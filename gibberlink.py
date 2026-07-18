"""
GibberLink Pro
--------------
Cambios respecto a la versión original (ver resumen al final del chat):

1. BUG DE DISTORSIÓN (el importante): ggwave-python entrega el audio ya
   codificado como PCM float32, no como int16. El código original leía
   esos bytes con dtype=np.int16, lo que interpretaba cada float32 como
   dos enteros de 16 bits sin sentido -> señal completamente corrupta.
   Lo mismo pasaba al revés en la recepción (se convertía a int16 antes
   de pasarlo a ggwave.decode, que espera float32). Esa es la causa de
   la distorsión: se ha corregido en transmitir_texto() y
   callback_receptor().
2. Mapeo de protocolos corregido a los IDs reales de ggwave y se ha
   quitado el "Pitch" simulado (remuestreo post-encode), que desplazaba
   las frecuencias y hacía indecodificable la ráfaga en el receptor.
   Ahora el modo "inaudible" usa protocolos ultrasónicos NATIVOS de
   ggwave, que sí son decodificables por otro receptor.
3. Ninguna actualización de Tkinter se hace ya desde hilos secundarios
   (pynput, hilo de grabación, hilo de recepción de audio): se usa
   set_status()/log() con root.after(), y las variables de la interfaz
   se cachean en atributos planos actualizados solo desde el hilo
   principal.
4. Selección de dispositivo de audio robustecida (ya no falla ni elige
   el último dispositivo silenciosamente si no hay nada seleccionado).
5. Cierre de la aplicación idempotente (ya no se libera ggwave ni se
   para pynput dos veces).
6. Lock para no encodear y decodear con la misma instancia de ggwave al
   mismo tiempo desde hilos distintos.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
import pyttsx3
import ggwave
import io
import time
from pynput import mouse


class GibberLinkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GibberLink Pro")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        # Inicializar Motores Base
        self.recognizer = sr.Recognizer()
        self.ggwave_instance = ggwave.init()

        # Evita encode/decode simultáneos sobre la misma instancia de ggwave
        self.ggwave_lock = threading.Lock()

        # Variables de Control de Hilos y Estado
        self.stream_receptor = None
        self.is_listening_stream = False
        self.is_ptt_pressed = False
        self.mouse_listener = None
        self._cerrado = False

        # Índices reales de sounddevice para los dispositivos seleccionados.
        # Se actualizan solo desde el hilo principal (eventos de Tkinter) y
        # se leen desde cualquier hilo, evitando tocar widgets fuera de su
        # hilo de origen.
        self.idx_in_actual = None
        self.idx_out_actual = None

        # Diccionario de Mapeo de Protocolos de GGwave (IDs reales de la
        # librería). 0-2 = audible, 3-4 = ultrasónico (inaudible de verdad).
        self.protocolos_map = {
            "Audible Normal (Recomendado)": 0,
            "Audible Rápido": 1,
            "Audible Ultra-Rápido (menos fiable)": 2,
            "Ultrasónico Normal (Inaudible)": 3,
            "Ultrasónico Rápido (Inaudible)": 4,
        }

        # Variables de Interfaz Controlables por el Usuario
        self.var_modo = tk.StringVar(value="Transmisión")
        self.var_protocolo = tk.StringVar(value="Audible Normal (Recomendado)")
        self.var_tts_speed = tk.IntVar(value=185)
        self.var_volumen = tk.DoubleVar(value=0.5)

        # Copias planas (no-Tkinter) de las variables anteriores, para
        # poder leerlas de forma segura desde hilos de fondo.
        self.modo_cache = self.var_modo.get()
        self.protocolo_cache = self.var_protocolo.get()
        self.tts_speed_cache = self.var_tts_speed.get()
        self.volumen_cache = self.var_volumen.get()

        self.var_modo.trace_add("write", lambda *a: setattr(self, "modo_cache", self.var_modo.get()))
        self.var_protocolo.trace_add("write", lambda *a: setattr(self, "protocolo_cache", self.var_protocolo.get()))
        self.var_tts_speed.trace_add("write", lambda *a: setattr(self, "tts_speed_cache", self.var_tts_speed.get()))
        self.var_volumen.trace_add("write", lambda *a: setattr(self, "volumen_cache", self.var_volumen.get()))

        self.dispositivos_input = []
        self.dispositivos_output = []

        self.crear_interfaz()
        self.cargar_dispositivos_audio()
        self.iniciar_captura_global_mouse()

    # ------------------------------------------------------------------
    # INTERFAZ
    # ------------------------------------------------------------------
    def crear_interfaz(self):
        frame_audio = ttk.LabelFrame(self.root, text=" 1. Enrutamiento de Tarjetas de Audio ", padding=10)
        frame_audio.pack(fill="x", padx=15, pady=5)

        ttk.Label(frame_audio, text="Entrada (Micrófono):").grid(row=0, column=0, sticky="w", pady=2)
        self.cb_input = ttk.Combobox(frame_audio, width=52, state="readonly")
        self.cb_input.grid(row=0, column=1, pady=2, padx=5)
        self.cb_input.bind("<<ComboboxSelected>>", self.actualizar_dispositivo_input)

        ttk.Label(frame_audio, text="Salida (Discord/Altavoz):").grid(row=1, column=0, sticky="w", pady=2)
        self.cb_output = ttk.Combobox(frame_audio, width=52, state="readonly")
        self.cb_output.grid(row=1, column=1, pady=2, padx=5)
        self.cb_output.bind("<<ComboboxSelected>>", self.actualizar_dispositivo_output)

        frame_parametros = ttk.LabelFrame(self.root, text=" 2. Ajustes del Modulador ", padding=10)
        frame_parametros.pack(fill="x", padx=15, pady=5)

        ttk.Label(frame_parametros, text="Protocolo de Modulación:").grid(row=0, column=0, sticky="w", pady=4)
        self.cb_protocolo = ttk.Combobox(frame_parametros, values=list(self.protocolos_map.keys()), textvariable=self.var_protocolo, width=40, state="readonly")
        self.cb_protocolo.grid(row=0, column=1, sticky="w", pady=4, padx=5)

        ttk.Label(frame_parametros, text="Velocidad lectura Voz (TTS):").grid(row=1, column=0, sticky="w", pady=4)
        self.slider_tts = ttk.Scale(frame_parametros, from_=100, to=300, variable=self.var_tts_speed, orient="horizontal", length=200)
        self.slider_tts.grid(row=1, column=1, sticky="w", pady=4, padx=5)
        self.lbl_tts_val = ttk.Label(frame_parametros, text="185 ppm")
        self.lbl_tts_val.grid(row=1, column=1, sticky="e", padx=5)
        self.var_tts_speed.trace_add("write", lambda *args: self.lbl_tts_val.config(text=f"{self.var_tts_speed.get()} ppm"))

        ttk.Label(frame_parametros, text="Multiplicador Volumen:").grid(row=2, column=0, sticky="w", pady=4)
        self.slider_vol = ttk.Scale(frame_parametros, from_=0.1, to=1.0, variable=self.var_volumen, orient="horizontal", length=200)
        self.slider_vol.grid(row=2, column=1, sticky="w", pady=4, padx=5)

        frame_modos = ttk.LabelFrame(self.root, text=" 3. Modo de Operación ", padding=10)
        frame_modos.pack(fill="x", padx=15, pady=5)

        modos = ["Transmisión", "Recepción", "Comunicación Completa"]
        for i, modo in enumerate(modos):
            rb = ttk.Radiobutton(frame_modos, text=modo, variable=self.var_modo, value=modo, command=self.actualizar_modo_operacion)
            rb.grid(row=0, column=i, padx=20, sticky="w")

        frame_monitor = ttk.LabelFrame(self.root, text=" Monitor de Actividad ", padding=10)
        frame_monitor.pack(fill="both", expand=True, padx=15, pady=5)

        self.lbl_status = ttk.Label(frame_monitor, text="ESTADO: Listo", font=("Helvetica", 11, "bold"))
        self.lbl_status.pack(anchor="w", pady=2)

        self.txt_log = scrolledtext.ScrolledText(frame_monitor, height=8, state="disabled", wrap="word")
        self.txt_log.pack(fill="both", expand=True, pady=5)

        lbl_ayuda = ttk.Label(self.root, text="Mantén presionado MOUSE 4 (Botón lateral) para transmitir.", font=("Helvetica", 9, "italic"))
        lbl_ayuda.pack(pady=5)

    # ------------------------------------------------------------------
    # LOG / ESTADO (siempre despachados al hilo principal)
    # ------------------------------------------------------------------
    def log(self, mensaje):
        def escribir():
            self.txt_log.configure(state="normal")
            self.txt_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {mensaje}\n")
            self.txt_log.see(tk.END)
            self.txt_log.configure(state="disabled")
        self.root.after(0, escribir)

    def set_status(self, texto, color="black"):
        self.root.after(0, lambda: self.lbl_status.config(text=texto, foreground=color))

    # ------------------------------------------------------------------
    # DISPOSITIVOS DE AUDIO
    # ------------------------------------------------------------------
    def cargar_dispositivos_audio(self):
        try:
            dispositivos = sd.query_devices()
            self.dispositivos_input = []
            self.dispositivos_output = []

            for i, d in enumerate(dispositivos):
                nombre = f"{i}: {d['name']}"
                if d['max_input_channels'] > 0:
                    self.dispositivos_input.append((nombre, i))
                if d['max_output_channels'] > 0:
                    self.dispositivos_output.append((nombre, i))

            self.cb_input['values'] = [d[0] for d in self.dispositivos_input]
            self.cb_output['values'] = [d[0] for d in self.dispositivos_output]

            if not self.dispositivos_input:
                self.log("Aviso: no se ha detectado ningún micrófono de entrada.")
            if not self.dispositivos_output:
                self.log("Aviso: no se ha detectado ningún dispositivo de salida.")

            try:
                def_in_index = sd.query_devices(kind='input')['index']
            except Exception:
                def_in_index = None
            try:
                def_out_index = sd.query_devices(kind='output')['index']
            except Exception:
                def_out_index = None

            seleccionado = False
            for idx, (nombre, i) in enumerate(self.dispositivos_input):
                if i == def_in_index:
                    self.cb_input.current(idx)
                    seleccionado = True
                    break
            if not seleccionado and self.dispositivos_input:
                self.cb_input.current(0)

            seleccionado = False
            for idx, (nombre, i) in enumerate(self.dispositivos_output):
                if i == def_out_index:
                    self.cb_output.current(idx)
                    seleccionado = True
                    break
            if not seleccionado and self.dispositivos_output:
                self.cb_output.current(0)

            # .current(idx) no dispara <<ComboboxSelected>>, así que
            # actualizamos el caché de índices manualmente.
            self.actualizar_dispositivo_input()
            self.actualizar_dispositivo_output()

        except Exception as e:
            self.log(f"Error cargando dispositivos de audio: {e}")

    def actualizar_dispositivo_input(self, event=None):
        idx = self.cb_input.current()
        if 0 <= idx < len(self.dispositivos_input):
            self.idx_in_actual = self.dispositivos_input[idx][1]

    def actualizar_dispositivo_output(self, event=None):
        idx = self.cb_output.current()
        if 0 <= idx < len(self.dispositivos_output):
            self.idx_out_actual = self.dispositivos_output[idx][1]

    # ------------------------------------------------------------------
    # TRANSMISIÓN (voz -> texto -> ráfaga ggwave)
    # ------------------------------------------------------------------
    def _quizas_reanudar_recepcion(self, modo_actual):
        if modo_actual == "Comunicación Completa":
            # Margen extra tras terminar de sonar el pitido, para dejar
            # que se apague el eco/reverberación de la sala antes de
            # volver a escuchar (evita autocapturar la propia ráfaga).
            self.root.after(500, self.iniciar_flujo_recepcion)

    def comenzar_grabacion_ptt(self):
        modo_actual = self.modo_cache
        idx_in = self.idx_in_actual
        idx_out = self.idx_out_actual
        if idx_in is None or idx_out is None:
            self.log("Error: no hay dispositivos de audio seleccionados.")
            self.set_status("ESTADO: Listo", "black")
            self._quizas_reanudar_recepcion(modo_actual)
            return

        self.log("Grabando... Habla ahora.")

        fs = 16000
        grabacion = []

        def callback_grabacion(indata, frames, time_info, status):
            if self.is_ptt_pressed:
                grabacion.append(indata.copy())

        try:
            with sd.InputStream(samplerate=fs, channels=1, device=idx_in, callback=callback_grabacion):
                while self.is_ptt_pressed:
                    sd.sleep(40)
        except Exception as e:
            self.log(f"Error abriendo micrófono: {e}")
            self.set_status("ESTADO: Listo", "black")
            self._quizas_reanudar_recepcion(modo_actual)
            return

        if len(grabacion) == 0:
            self.set_status("ESTADO: Listo", "black")
            self._quizas_reanudar_recepcion(modo_actual)
            return

        audio_completo = np.concatenate(grabacion, axis=0)

        wav_io = io.BytesIO()
        wav.write(wav_io, fs, (audio_completo * 32767).astype(np.int16))
        wav_io.seek(0)

        self.log("Procesando Voz a Texto...")
        try:
            with sr.AudioFile(wav_io) as source:
                audio_data = self.recognizer.record(source)

            texto = self.recognizer.recognize_google(audio_data, language="es-ES")
            self.log(f"Texto procesado: '{texto}'")

            self.transmitir_texto(texto, idx_out)

        except sr.UnknownValueError:
            self.log("Aviso: No se detectó una voz clara.")
        except sr.RequestError as e:
            self.log(f"Error del servicio de reconocimiento de voz (¿sin conexión a internet?): {e}")
        except Exception as e:
            self.log(f"Error en transmisión: {e}")
        finally:
            self.set_status("ESTADO: Listo", "black")
            self._quizas_reanudar_recepcion(modo_actual)

    def transmitir_texto(self, texto, idx_out):
        payload = texto.encode('utf-8')
        proto_id = self.protocolos_map.get(self.protocolo_cache, 0)

        with self.ggwave_lock:
            waveform = ggwave.encode(payload, protocolId=proto_id, volume=20, instance=self.ggwave_instance)

        raw_bytes = bytes(waveform)

        # *** FIX PRINCIPAL DE LA DISTORSIÓN ***
        # ggwave-python devuelve el audio como PCM float32, NO como int16.
        # Leerlo como int16 (como hacía la versión anterior) corrompe la
        # señal por completo.
        audio_float = np.frombuffer(raw_bytes, dtype=np.float32).copy()

        if len(audio_float) == 0:
            self.log("Error: la ráfaga generada está vacía.")
            return

        # Recorte de silencio residual al final de la ráfaga
        indices_con_sonido = np.where(np.abs(audio_float) > 0.001)[0]
        if len(indices_con_sonido) > 0:
            audio_float = audio_float[:indices_con_sonido[-1] + 400]

        # Normalizamos a un pico de referencia conocido
        max_pico = np.max(np.abs(audio_float))
        if max_pico > 0:
            audio_float = audio_float / max_pico

        # Ganancia controlada por el usuario + limitador suave (tanh) en
        # vez de un recorte duro: evita introducir armónicos que
        # dificultarían la decodificación en el receptor.
        vol_factor = self.volumen_cache
        audio_final = audio_float * vol_factor
        audio_final = np.tanh(audio_final / 0.95) * 0.95
        audio_final = audio_final.astype(np.float32)

        es_ultrasonico = proto_id >= 3
        etiqueta = "ultrasónica (inaudible)" if es_ultrasonico else "audible"
        self.log(f"Disparando ráfaga {etiqueta} a 48000Hz...")

        try:
            # Duplicamos la señal mono a todos los canales disponibles del
            # dispositivo de salida (normalmente L+R), así suena por los
            # dos altavoces en vez de solo por uno.
            try:
                canales_out = sd.query_devices(idx_out)['max_output_channels']
            except Exception:
                canales_out = 1

            if canales_out >= 2:
                audio_salida = np.column_stack([audio_final] * 2)
            else:
                audio_salida = audio_final

            sd.play(audio_salida, samplerate=48000, device=idx_out)
            sd.wait()
            self.log("Ráfaga finalizada.")
        except Exception as e:
            self.log(f"Error reproduciendo la ráfaga: {e}")

    # ------------------------------------------------------------------
    # RECEPCIÓN (ráfaga ggwave -> texto -> TTS)
    # ------------------------------------------------------------------
    def iniciar_flujo_recepcion(self):
        if self.is_listening_stream:
            return
        idx_in = self.idx_in_actual
        if idx_in is None:
            self.log("Error: no hay micrófono seleccionado para recepción.")
            return
        try:
            self.stream_receptor = sd.InputStream(
                samplerate=48000,
                channels=1,
                device=idx_in,
                callback=self.callback_receptor,
                blocksize=1024
            )
            self.stream_receptor.start()
            self.is_listening_stream = True
            self.log("Receptor escuchando el espectro de audio...")
        except Exception as e:
            self.log(f"Error abriendo receptor: {e}")

    def detener_flujo_recepcion(self):
        if not self.is_listening_stream:
            return
        try:
            if self.stream_receptor:
                self.stream_receptor.stop()
                self.stream_receptor.close()
            self.is_listening_stream = False
            self.log("Receptor apagado.")
        except Exception as e:
            self.log(f"Error al apagar el receptor: {e}")

    def callback_receptor(self, indata, frames, time_info, status):
        # ggwave.decode espera PCM float32 en bruto, igual que llega desde
        # sounddevice. Convertir antes a int16 (como hacía la versión
        # anterior) rompía la decodificación de forma silenciosa.
        audio_data = np.ascontiguousarray(indata[:, 0], dtype=np.float32).tobytes()

        with self.ggwave_lock:
            res = ggwave.decode(self.ggwave_instance, audio_data)

        if res:
            try:
                texto_decodificado = res.decode('utf-8')
                self.log(f"¡Ráfaga capturada!: {texto_decodificado}")

                vel_lectura = self.tts_speed_cache
                threading.Thread(target=self.ejecutar_tts, args=(texto_decodificado, vel_lectura), daemon=True).start()
            except Exception as e:
                self.log(f"Error decodificando bytes: {e}")

    def ejecutar_tts(self, texto, velocidad):
        try:
            engine_local = pyttsx3.init()
            engine_local.setProperty('rate', velocidad)
            engine_local.say(texto)
            engine_local.runAndWait()
        except Exception as e:
            self.log(f"Fallo en síntesis TTS: {e}")

    # ------------------------------------------------------------------
    # CONTROLADORES DE EVENTOS
    # ------------------------------------------------------------------
    def actualizar_modo_operacion(self):
        # Se llama desde el hilo principal (command= de un Radiobutton),
        # así que leer la StringVar aquí es seguro.
        modo = self.var_modo.get()
        self.log(f"Cambiando a modo: [{modo}]")
        if modo == "Transmisión":
            self.detener_flujo_recepcion()
        elif modo in ["Recepción", "Comunicación Completa"]:
            self.iniciar_flujo_recepcion()

    def on_click(self, x, y, button, pressed):
        # Se ejecuta en el hilo del listener global de pynput: solo se
        # tocan cachés planos y métodos ya protegidos con root.after().
        if button == mouse.Button.x1:
            modo_actual = self.modo_cache
            if modo_actual not in ["Transmisión", "Comunicación Completa"]:
                return

            if pressed and not self.is_ptt_pressed:
                self.is_ptt_pressed = True
                self.set_status("ESTADO: CAPTURANDO MICRO...", "red")
                if modo_actual == "Comunicación Completa":
                    self.detener_flujo_recepcion()

                threading.Thread(target=self.comenzar_grabacion_ptt, daemon=True).start()

            elif not pressed and self.is_ptt_pressed:
                self.is_ptt_pressed = False
                self.set_status("ESTADO: ENVIANDO PITIDO...", "orange")
                # La recepción se reactiva sola cuando termina TODA la
                # transmisión real (STT + codificación + reproducción),
                # ver comenzar_grabacion_ptt(). Reactivarla aquí con un
                # temporizador fijo hacía que el receptor volviera a
                # escuchar antes de que terminara de sonar el propio
                # pitido, y se autocapturaba por eco acústico.

    def iniciar_captura_global_mouse(self):
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()

    # ------------------------------------------------------------------
    # CIERRE (idempotente: seguro llamarlo más de una vez)
    # ------------------------------------------------------------------
    def cerrar(self):
        if self._cerrado:
            return
        self._cerrado = True
        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
        except Exception:
            pass
        self.detener_flujo_recepcion()
        try:
            ggwave.free(self.ggwave_instance)
        except Exception:
            pass

    def __del__(self):
        try:
            self.cerrar()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = GibberLinkApp(root)

    def al_cerrar():
        app.cerrar()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", al_cerrar)
    root.mainloop()