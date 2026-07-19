"""GibberLink Pro: escritorio PySide6 para mensajes acústicos con GGWave.

Cambios sobre la versión anterior (hecha con Codex):

BUGS ARREGLADOS
1. El modo guardado ("Recepción"/"Comunicación completa") no arrancaba el
   receptor al abrir la app: build_ui() marcaba el radio/botón antes de
   que connect_signals() conectara el evento. Ahora se fuerza
   actualizar_modo_operacion(self.modo_cache) explícitamente tras cargar
   dispositivos, así que el modo guardado se aplica de verdad.
2. El ACK automático se transmitía sin pausar el receptor, así que era
   fácil que se autocapturara su propio "ACK:OK". Ahora enviar_ack()
   pausa y reanuda igual que el resto de envíos.
3. QtCharts y QtWebEngine son módulos opcionales de PySide6 que pueden no
   estar instalados; si faltaban, la app entera no arrancaba. Ahora los
   imports están protegidos y esas pestañas degradan a una versión
   simple (medidor con QProgressBar) o a un aviso, en vez de reventar.
4. El medidor de nivel emitía en cada bloque de audio (~47 veces/seg)
   aunque la pestaña de Diagnóstico no estuviera visible. Ahora se
   diezma (1 de cada 4 bloques).
5. El límite de 140 bytes UTF-8 solo se comprobaba en el mensaje escrito
   a mano, no en el transcrito por voz. Ahora usa la misma validación.
6. apply_glass_effects() se llamaba antes de crear las pestañas de
   Ajustes/Diagnóstico/Visualizador, así que esas tarjetas se quedaban
   sin la sombra de "cristal". Ahora se aplica al final, a todo.
7. Índices de pestaña hardcodeados (Ctrl+D, abrir_visualizador_web) que
   se habrían roto al reordenar pestañas: ahora se guardan referencias
   directas a cada página en vez de índices numéricos.

CAMBIOS DE DISEÑO/UX PEDIDOS
- El enrutamiento de audio y los ajustes de señal/voz se han movido de
  la pestaña "Operador" a la pestaña "Ajustes", dejando el Operador
  centrado solo en enviar/recibir y monitorizar.
- El modo de operación ahora es un interruptor compacto tipo "pastilla"
  con Tx / Rx / Tx/Rx, en vez de tres radiobuttons en una caja entera.
- Paleta con acentos de color (violeta/cian) en vez de solo grises,
  título con icono, y un pequeño "pulso" de opacidad en el estado cada
  vez que cambia, para que se note más vivo.
"""

import io
import json
import threading
import time
from pathlib import Path

import ggwave
import numpy as np
import pyttsx3
import scipy.io.wavfile as wav
import sounddevice as sd
import speech_recognition as sr
from pynput import mouse
from PySide6.QtCore import QEvent, QPointF, QPropertyAnimation, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QPlainTextEdit, QProgressBar, QPushButton,
    QSlider, QSplitter, QStyle, QSystemTrayIcon, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

# QtCharts y QtWebEngine son paquetes ADICIONALES de PySide6 (no siempre
# vienen con "pip install PySide6"). Si faltan, la app debe seguir
# funcionando con esas dos pestañas en modo degradado, no reventar.
try:
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    QTCHARTS_DISPONIBLE = True
except ImportError:
    QTCHARTS_DISPONIBLE = False

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    QTWEBENGINE_DISPONIBLE = True
except ImportError:
    QTWEBENGINE_DISPONIBLE = False


class GibberLinkApp(QMainWindow):
    """La UI vive en Qt; audio, STT y TTS se ejecutan fuera del hilo visual."""

    log_signal = Signal(str)
    status_signal = Signal(str, str)
    history_signal = Signal(str, str)
    signal_info = Signal(str, float, int)
    meter_signal = Signal(float)
    notification_signal = Signal(str, str)
    resume_signal = Signal()
    send_finished = Signal()

    PROTOCOLOS = {
        "Audible normal (recomendado)": 0,
        "Audible rápido": 1,
        "Audible ultrarrápido (menos fiable)": 2,
        "Ultrasónico normal (inaudible)": 3,
        "Ultrasónico rápido (inaudible)": 4,
    }
    IDIOMAS = {
        "Español (España)": "es-ES", "English (US)": "en-US",
        "Français": "fr-FR", "Deutsch": "de-DE",
    }
    MODOS = [
        ("Transmisión", "Tx"),
        ("Recepción", "Rx"),
        ("Comunicación completa", "Tx/Rx"),
    ]

    def __init__(self):
        super().__init__()
        self.config_path = Path(__file__).with_name("gibberlink_config.json")
        self.preferences = self.load_preferences()
        self.recognizer = sr.Recognizer()
        self.ggwave_instance = ggwave.init()
        self.ggwave_lock = threading.Lock()
        self.stream_receptor = None
        self.is_listening_stream = False
        self.is_ptt_pressed = False
        self.mouse_listener = None
        self.closed = False
        self.idx_in_actual = None
        self.idx_out_actual = None
        self.dispositivos_input, self.dispositivos_output = [], []
        self.ultimo_texto_enviado = ""
        self.modo_cache = self.preferences.get("modo", "Transmisión")
        self.protocolo_cache = self.preferences.get("protocolo", "Audible normal (recomendado)")
        self.volumen_cache = float(self.preferences.get("volumen", 0.5))
        self.tts_speed_cache = int(self.preferences.get("tts_speed", 185))
        self.idioma_cache = self.preferences.get("idioma", "Español (España)")
        self.confirmar_cache = bool(self.preferences.get("confirmar", True))
        self.notificar_cache = bool(self.preferences.get("notificaciones", True))
        self.settings = QSettings("GibberLink", "GibberLink Pro")
        self.meter_values = [0.0] * 60
        self._meter_counter = 0
        self._status_anim = None
        self.shortcuts = []

        self.setWindowTitle("GibberLink Pro")
        self.resize(980, 760)
        self.setMinimumSize(800, 630)
        self.build_ui()
        self.connect_signals()
        self.cargar_dispositivos_audio()
        # FIX: aplica de verdad el modo guardado (antes se quedaba solo
        # visualmente marcado sin arrancar el receptor).
        self.actualizar_modo_operacion(self.modo_cache)
        self.iniciar_captura_global_mouse()
        self.crear_bandeja_sistema()
        self.configurar_atajos()
        if self.settings.value("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))

    # ---------- UI ----------
    def build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.crear_pestana_operador()
        self.create_settings_tab()
        self.create_diagnostics_tab()
        self.create_visualizer_tab()

        # Se aplica al final, ya con TODAS las pestañas construidas, para
        # que las tarjetas de Ajustes/Diagnóstico también tengan sombra.
        self.setStyleSheet(STYLESHEET)
        self.apply_glass_effects()

    def crear_pestana_operador(self):
        page = QWidget()
        self.pagina_operador = page
        self.tabs.addTab(page, "Operador")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        header_row = QHBoxLayout()
        title = QLabel("📡 GibberLink")
        title.setObjectName("title")
        header_row.addWidget(title)
        subtitle = QLabel("  ·  consola de señales acústicas")
        subtitle.setObjectName("subtitle")
        header_row.addWidget(subtitle)
        header_row.addStretch()
        header_row.addWidget(self.crear_selector_modo())
        layout.addLayout(header_row)

        message_box = QGroupBox("Enviar mensaje directo")
        message_layout = QVBoxLayout(message_box)
        message_layout.addWidget(QLabel("Convierte texto en señal sin usar el micrófono."))
        send_row = QHBoxLayout()
        self.message_edit = QPlainTextEdit()
        self.message_edit.setPlaceholderText("Escribe un mensaje (Ctrl + Intro para enviar)")
        self.message_edit.setFixedHeight(78)
        send_row.addWidget(self.message_edit, 1)
        self.send_button = QPushButton("Enviar señal  ↗")
        self.send_button.setObjectName("accent")
        self.send_button.setMinimumWidth(145)
        send_row.addWidget(self.send_button)
        message_layout.addLayout(send_row)
        action_row = QHBoxLayout()
        self.repeat_button = QPushButton("Repetir último")
        self.repeat_button.setEnabled(False)
        action_row.addWidget(self.repeat_button)
        action_row.addStretch()
        self.signal_label = QLabel("Señal: esperando un envío")
        self.signal_label.setObjectName("muted")
        action_row.addWidget(self.signal_label)
        message_layout.addLayout(action_row)
        self.signal_bar = QProgressBar()
        self.signal_bar.setRange(0, 100)
        self.signal_bar.setValue(0)
        self.signal_bar.setTextVisible(False)
        message_layout.addWidget(self.signal_bar)
        layout.addWidget(message_box)

        monitor = QGroupBox("Monitor de actividad")
        monitor_layout = QVBoxLayout(monitor)
        self.status_label = QLabel("● ESTADO: Listo")
        self.status_label.setObjectName("status")
        self.status_label.setStyleSheet("color: #86efac;")
        self.status_effect = QGraphicsOpacityEffect(self.status_label)
        self.status_label.setGraphicsEffect(self.status_effect)
        monitor_layout.addWidget(self.status_label)
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Hora", "Tipo", "Mensaje"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setMaximumHeight(160)
        self.history_table.setColumnWidth(0, 75)
        self.history_table.setColumnWidth(1, 105)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        monitor_layout.addWidget(self.history_table)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(300)
        self.log_edit.setFixedHeight(92)
        monitor_layout.addWidget(self.log_edit)
        layout.addWidget(monitor, 1)

        hint = QLabel("Mantén MOUSE 4 (botón lateral) para enviar voz.")
        hint.setObjectName("muted")
        layout.addWidget(hint)

    def crear_selector_modo(self):
        """Interruptor compacto tipo 'pastilla': Tx / Rx / Tx-Rx."""
        frame = QFrame()
        frame.setObjectName("modeSwitch")
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(3, 3, 3, 3)
        frame_layout.setSpacing(0)

        self.mode_buttons = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        for valor, etiqueta in self.MODOS:
            boton = QPushButton(etiqueta)
            boton.setObjectName("modeSeg")
            boton.setCheckable(True)
            boton.setChecked(valor == self.modo_cache)
            boton.setCursor(Qt.CursorShape.PointingHandCursor)
            boton.setToolTip(valor)
            self.mode_group.addButton(boton)
            self.mode_buttons[valor] = boton
            frame_layout.addWidget(boton)

        return frame

    def apply_glass_effects(self):
        """Profundidad suave para simular capas de cristal sobre el fondo oscuro."""
        for card in self.findChildren(QGroupBox):
            shadow = QGraphicsDropShadowEffect(card)
            shadow.setBlurRadius(28)
            shadow.setOffset(0, 8)
            shadow.setColor(QColor(0, 0, 0, 105))
            card.setGraphicsEffect(shadow)
        for frame in self.findChildren(QFrame):
            if frame.objectName() == "modeSwitch":
                shadow = QGraphicsDropShadowEffect(frame)
                shadow.setBlurRadius(20)
                shadow.setOffset(0, 6)
                shadow.setColor(QColor(0, 0, 0, 110))
                frame.setGraphicsEffect(shadow)
        for button in self.findChildren(QPushButton):
            if button.objectName() == "modeSeg":
                continue  # ya llevan sombra de conjunto vía el QFrame
            shadow = QGraphicsDropShadowEffect(button)
            shadow.setBlurRadius(16)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(0, 0, 0, 90))
            button.setGraphicsEffect(shadow)

    def create_diagnostics_tab(self):
        page = QWidget()
        self.pagina_diagnostico = page
        self.tabs.addTab(page, "Diagnóstico")
        layout = QVBoxLayout(page)
        summary = QLabel("Nivel de entrada en tiempo real. Úsalo para comprobar micrófono, eco y ganancia antes de transmitir.")
        summary.setObjectName("muted")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        if QTCHARTS_DISPONIBLE:
            splitter = QSplitter(Qt.Orientation.Vertical)
            chart = QChart()
            chart.setBackgroundVisible(False)
            chart.legend().hide()
            self.meter_series = QLineSeries()
            chart.addSeries(self.meter_series)
            axis_x = QValueAxis()
            axis_x.setRange(0, 59)
            axis_x.setLabelsVisible(False)
            axis_y = QValueAxis()
            axis_y.setRange(0, 1)
            axis_y.setTitleText("Nivel")
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            self.meter_series.attachAxis(axis_x)
            self.meter_series.attachAxis(axis_y)
            self.meter_chart = QChartView(chart)
            splitter.addWidget(self.meter_chart)
        else:
            aviso = QLabel(
                "Gráfico detallado no disponible (falta el módulo opcional "
                "PySide6-Addons/QtCharts). Se muestra un medidor simple."
            )
            aviso.setObjectName("muted")
            aviso.setWordWrap(True)
            self.meter_bar = QProgressBar()
            self.meter_bar.setRange(0, 100)
            self.meter_bar.setTextVisible(True)
            contenedor = QWidget()
            cont_layout = QVBoxLayout(contenedor)
            cont_layout.addWidget(aviso)
            cont_layout.addWidget(self.meter_bar)
            cont_layout.addStretch()

        diagnostic_box = QGroupBox("Comprobaciones rápidas")
        diagnostic_layout = QVBoxLayout(diagnostic_box)
        self.diag_label = QLabel("Esperando datos del micrófono.")
        diagnostic_layout.addWidget(self.diag_label)
        test_button = QPushButton("Abrir prueba guiada de audio")
        test_button.clicked.connect(self.mostrar_prueba_audio)
        diagnostic_layout.addWidget(test_button)

        if QTCHARTS_DISPONIBLE:
            splitter.addWidget(diagnostic_box)
            splitter.setSizes([300, 130])
            layout.addWidget(splitter)
        else:
            layout.addWidget(contenedor)
            layout.addWidget(diagnostic_box)

    def create_visualizer_tab(self):
        page = QWidget()
        self.pagina_visualizador = page
        self.tabs.addTab(page, "Visualizador")
        layout = QVBoxLayout(page)

        if not QTWEBENGINE_DISPONIBLE:
            aviso = QLabel(
                "El visualizador web no está disponible: falta el módulo opcional "
                "PySide6-WebEngine.\n\nPuedes instalarlo con:\n"
                "pip install PySide6-Addons"
            )
            aviso.setWordWrap(True)
            aviso.setObjectName("muted")
            layout.addWidget(aviso)
            layout.addStretch()
            return

        label = QLabel("Visualizador web: una capa HTML/CSS para experimentar con gráficos y efectos que Qt nativo no ofrece.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.visualizer_placeholder = QPushButton("Cargar visualizador líquido")
        self.visualizer_placeholder.clicked.connect(self.abrir_visualizador_web)
        layout.addWidget(self.visualizer_placeholder)
        layout.addStretch()

    def create_settings_tab(self):
        page = QWidget()
        self.pagina_ajustes = page
        self.tabs.addTab(page, "Ajustes")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        audio_box = QGroupBox("Enrutamiento de audio")
        audio_form = QFormLayout(audio_box)
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        audio_form.addRow("Entrada (micrófono):", self.input_combo)
        audio_form.addRow("Salida (altavoz/Discord):", self.output_combo)
        layout.addWidget(audio_box)

        settings_box = QGroupBox("Señal y voz")
        settings_form = QFormLayout(settings_box)
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(self.PROTOCOLOS)
        self.protocol_combo.setCurrentText(self.protocolo_cache)
        self.language_combo = QComboBox()
        self.language_combo.addItems(self.IDIOMAS)
        self.language_combo.setCurrentText(self.idioma_cache)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(10, 100)
        self.volume_slider.setValue(round(self.volumen_cache * 100))
        self.tts_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_slider.setRange(100, 300)
        self.tts_slider.setValue(self.tts_speed_cache)
        self.ack_check = QCheckBox("Confirmación automática (ACK)")
        self.ack_check.setChecked(self.confirmar_cache)
        settings_form.addRow("Protocolo:", self.protocol_combo)
        settings_form.addRow("Idioma de voz:", self.language_combo)
        settings_form.addRow("Volumen:", self.volume_slider)
        settings_form.addRow("Velocidad TTS:", self.tts_slider)
        settings_form.addRow("", self.ack_check)
        layout.addWidget(settings_box)

        desktop_box = QGroupBox("Experiencia de escritorio")
        form = QFormLayout(desktop_box)
        self.notification_check = QCheckBox("Mostrar avisos en bandeja del sistema")
        self.notification_check.setChecked(self.notificar_cache)
        self.sound_check = QCheckBox("Emitir sonido de aviso")
        self.sound_check.setChecked(True)
        test_button = QPushButton("Probar configuración de audio")
        test_button.clicked.connect(self.mostrar_prueba_audio)
        form.addRow("Notificaciones:", self.notification_check)
        form.addRow("Sonido:", self.sound_check)
        form.addRow("Asistente:", test_button)
        layout.addWidget(desktop_box)

        shortcuts = QGroupBox("Atajos")
        shortcut_layout = QFormLayout(shortcuts)
        shortcut_layout.addRow("Ctrl + Intro", QLabel("Enviar mensaje"))
        shortcut_layout.addRow("Ctrl + R", QLabel("Repetir último mensaje"))
        shortcut_layout.addRow("Ctrl + D", QLabel("Abrir diagnóstico"))
        shortcut_layout.addRow("Ctrl + M", QLabel("Cambiar entre transmisión y recepción"))
        layout.addWidget(shortcuts)
        layout.addStretch()

    def abrir_visualizador_web(self):
        if hasattr(self, "web_view"):
            return
        self.web_view = QWebEngineView()
        self.pagina_visualizador.layout().replaceWidget(self.visualizer_placeholder, self.web_view)
        self.visualizer_placeholder.deleteLater()
        self.web_view.setHtml("""
            <style>body{margin:0;background:#111;color:#eee;font:16px Segoe UI;display:grid;place-items:center;height:100vh;overflow:hidden}
            .orb{width:42vmin;height:42vmin;border-radius:50%;background:radial-gradient(circle at 30% 25%,#fff,#777 28%,#222 66%);box-shadow:0 0 120px #aaa8,inset -30px -30px 60px #000;animation:float 4s ease-in-out infinite}
            @keyframes float{50%{transform:translateY(-20px) scale(1.05);filter:brightness(1.2)}}</style><div class='orb'></div><p>Señal acústica · visualizador experimental</p>
        """)

    def mostrar_prueba_audio(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Prueba guiada de audio")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("1. Comprueba que el nivel del micrófono se mueve en Diagnóstico.\n2. Envía un mensaje corto con volumen al 50%.\n3. Si hay eco, baja volumen o separa micrófono y altavoz."))
        state = "listos" if self.idx_in_actual is not None and self.idx_out_actual is not None else "incompletos"
        layout.addWidget(QLabel(f"Dispositivos: {state}."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def crear_bandeja_sistema(self):
        self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), self)
        menu = QMenu(self)
        show_action = QAction("Mostrar GibberLink", self)
        show_action.triggered.connect(self.showNormal)
        quit_action = QAction("Salir", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("GibberLink Pro")
        self.tray.show()

    def configurar_atajos(self):
        for sequence, callback in (
            ("Ctrl+R", self.repetir_ultimo_mensaje),
            ("Ctrl+D", lambda: self.tabs.setCurrentWidget(self.pagina_diagnostico)),
            ("Ctrl+M", self.alternar_modo),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def alternar_modo(self):
        target = "Recepción" if self.modo_cache == "Transmisión" else "Transmisión"
        self.mode_buttons[target].setChecked(True)

    def connect_signals(self):
        self.send_button.clicked.connect(self.enviar_mensaje_escrito)
        self.repeat_button.clicked.connect(self.repetir_ultimo_mensaje)
        self.message_edit.installEventFilter(self)
        self.input_combo.currentIndexChanged.connect(self.actualizar_dispositivo_input)
        self.output_combo.currentIndexChanged.connect(self.actualizar_dispositivo_output)
        self.protocol_combo.currentTextChanged.connect(lambda value: setattr(self, "protocolo_cache", value))
        self.language_combo.currentTextChanged.connect(lambda value: setattr(self, "idioma_cache", value))
        self.volume_slider.valueChanged.connect(lambda value: setattr(self, "volumen_cache", value / 100))
        self.tts_slider.valueChanged.connect(lambda value: setattr(self, "tts_speed_cache", value))
        self.ack_check.toggled.connect(lambda value: setattr(self, "confirmar_cache", value))
        for mode, button in self.mode_buttons.items():
            button.toggled.connect(lambda checked, value=mode: checked and self.actualizar_modo_operacion(value))
        self.log_signal.connect(self.append_log)
        self.status_signal.connect(self.update_status)
        self.history_signal.connect(self.add_history)
        self.signal_info.connect(self.update_signal_info)
        self.meter_signal.connect(self.update_meter)
        self.notification_signal.connect(self.mostrar_notificacion)
        self.history_table.customContextMenuRequested.connect(self.mostrar_menu_historial)
        self.notification_check.toggled.connect(lambda value: setattr(self, "notificar_cache", value))
        self.resume_signal.connect(lambda: QTimer.singleShot(500, self.iniciar_flujo_recepcion))
        self.send_finished.connect(lambda: self.send_button.setEnabled(True))

    def eventFilter(self, source, event):
        if source is self.message_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.enviar_mensaje_escrito()
                return True
        return super().eventFilter(source, event)

    # ---------- Presentación segura para hilos ----------
    def log(self, message): self.log_signal.emit(message)
    def set_status(self, text, color="#86efac"): self.status_signal.emit(text, color)
    def add_history_threadsafe(self, kind, message): self.history_signal.emit(kind, message)

    def append_log(self, message):
        self.log_edit.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def update_status(self, text, color):
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(f"color: {color};")
        # Pequeño "pulso" de opacidad cada vez que cambia el estado, para
        # que se note el cambio en vez de un simple salto de texto.
        anim = QPropertyAnimation(self.status_effect, b"opacity", self)
        anim.setDuration(280)
        anim.setStartValue(0.25)
        anim.setEndValue(1.0)
        anim.start()
        self._status_anim = anim  # evita que el GC se lo lleve a medias

    def add_history(self, kind, message):
        self.history_table.insertRow(0)
        for col, value in enumerate((time.strftime("%H:%M:%S"), kind, message)):
            self.history_table.setItem(0, col, QTableWidgetItem(value))
        while self.history_table.rowCount() > 100:
            self.history_table.removeRow(self.history_table.rowCount() - 1)

    def update_signal_info(self, protocol, duration, power):
        self.signal_label.setText(f"Señal: {protocol} · {duration:.2f} s · pico {power}%")
        self.signal_animation = QPropertyAnimation(self.signal_bar, b"value", self)
        self.signal_animation.setDuration(420)
        self.signal_animation.setStartValue(self.signal_bar.value())
        self.signal_animation.setEndValue(power)
        self.signal_animation.start()

    def update_meter(self, level):
        self.meter_values = self.meter_values[1:] + [level]
        if QTCHARTS_DISPONIBLE and hasattr(self, "meter_series"):
            self.meter_series.replace([QPointF(index, value) for index, value in enumerate(self.meter_values)])
        elif hasattr(self, "meter_bar"):
            self.meter_bar.setValue(int(level * 100))
        self.diag_label.setText(f"Nivel de entrada: {int(level * 100)}% · {'Señal saludable' if level > .03 else 'Sin actividad'}")

    def mostrar_menu_historial(self, position):
        row = self.history_table.rowAt(position.y())
        if row < 0:
            return
        menu = QMenu(self)
        copy_action = menu.addAction("Copiar mensaje")
        resend_action = menu.addAction("Reenviar este mensaje")
        chosen = menu.exec(self.history_table.viewport().mapToGlobal(position))
        message_item = self.history_table.item(row, 2)
        if not message_item:
            return
        if chosen == copy_action:
            QApplication.clipboard().setText(message_item.text())
        elif chosen == resend_action:
            self.message_edit.setPlainText(message_item.text())
            self.tabs.setCurrentWidget(self.pagina_operador)

    def mostrar_notificacion(self, title, message):
        if self.notificar_cache and self.tray.isVisible():
            self.tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 3500)
        if self.sound_check.isChecked():
            QApplication.beep()

    # ---------- Preferencias y dispositivos ----------
    def load_preferences(self):
        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_preferences(self):
        data = {
            "modo": self.modo_cache, "protocolo": self.protocolo_cache,
            "volumen": self.volumen_cache, "tts_speed": self.tts_speed_cache,
            "idioma": self.idioma_cache, "confirmar": self.confirmar_cache, "notificaciones": self.notificar_cache,
            "input": self.input_combo.currentText(), "output": self.output_combo.currentText(),
        }
        try:
            with self.config_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
        except OSError as error:
            self.log(f"No se pudieron guardar preferencias: {error}")

    def cargar_dispositivos_audio(self):
        try:
            devices = sd.query_devices()
            self.dispositivos_input = [(f"{i}: {d['name']}", i) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
            self.dispositivos_output = [(f"{i}: {d['name']}", i) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
            self.input_combo.addItems(name for name, _ in self.dispositivos_input)
            self.output_combo.addItems(name for name, _ in self.dispositivos_output)
            self.select_saved_device(self.input_combo, self.dispositivos_input, self.preferences.get("input"))
            self.select_saved_device(self.output_combo, self.dispositivos_output, self.preferences.get("output"))
            self.actualizar_dispositivo_input()
            self.actualizar_dispositivo_output()
        except Exception as error:
            self.log(f"Error cargando dispositivos de audio: {error}")

    @staticmethod
    def select_saved_device(combo, devices, saved):
        index = next((i for i, (name, _) in enumerate(devices) if name == saved), 0)
        if devices:
            combo.setCurrentIndex(index)

    def actualizar_dispositivo_input(self):
        index = self.input_combo.currentIndex()
        self.idx_in_actual = self.dispositivos_input[index][1] if index >= 0 else None

    def actualizar_dispositivo_output(self):
        index = self.output_combo.currentIndex()
        self.idx_out_actual = self.dispositivos_output[index][1] if index >= 0 else None

    # ---------- Validación ----------
    def validar_longitud_mensaje(self, texto):
        tam = len(texto.encode("utf-8"))
        if tam > 140:
            self.log(f"Aviso: mensaje demasiado largo ({tam} bytes, máximo 140 UTF-8) — no se ha enviado.")
            return False
        return True

    # ---------- Envío ----------
    def enviar_mensaje_escrito(self):
        text = self.message_edit.toPlainText().strip()
        if not text:
            self.log("Escribe un mensaje antes de enviarlo.")
            return
        if not self.validar_longitud_mensaje(text):
            return
        if self.idx_out_actual is None:
            self.log("No hay dispositivo de salida seleccionado.")
            return
        if self.is_listening_stream:
            self.detener_flujo_recepcion()
        self.ultimo_texto_enviado = text
        self.repeat_button.setEnabled(True)
        self.send_button.setEnabled(False)
        self.set_status("ESTADO: ENVIANDO MENSAJE…", "#fbbf24")
        self.add_history_threadsafe("Enviado", text)
        self.log(f"Enviando mensaje escrito: '{text}'")
        threading.Thread(target=self.enviar_texto_worker, args=(text, self.idx_out_actual, self.modo_cache), daemon=True).start()

    def repetir_ultimo_mensaje(self):
        if self.ultimo_texto_enviado:
            self.message_edit.setPlainText(self.ultimo_texto_enviado)
            self.enviar_mensaje_escrito()

    def enviar_texto_worker(self, text, output_index, mode):
        try:
            self.transmitir_texto(text, output_index)
        finally:
            self.send_finished.emit()
            self.set_status("ESTADO: Listo")
            if mode == "Comunicación completa":
                self.resume_signal.emit()

    def transmitir_texto(self, text, output_index):
        protocol_id = self.PROTOCOLOS.get(self.protocolo_cache, 0)
        with self.ggwave_lock:
            waveform = ggwave.encode(text.encode("utf-8"), protocolId=protocol_id, volume=20, instance=self.ggwave_instance)
        audio = np.frombuffer(bytes(waveform), dtype=np.float32).copy()
        if not len(audio):
            self.log("Error: la ráfaga generada está vacía.")
            return
        sound = np.where(np.abs(audio) > 0.001)[0]
        if len(sound):
            audio = audio[:sound[-1] + 400]
        peak = np.max(np.abs(audio))
        if peak:
            audio /= peak
        audio = (np.tanh((audio * self.volumen_cache) / .95) * .95).astype(np.float32)
        self.signal_info.emit(self.protocolo_cache, len(audio) / 48000, int(min(100, np.max(np.abs(audio)) * 100)))
        try:
            channels = sd.query_devices(output_index)["max_output_channels"]
            output = np.column_stack([audio, audio]) if channels >= 2 else audio
            sd.play(output, samplerate=48000, device=output_index)
            sd.wait()
            self.log("Ráfaga finalizada.")
        except Exception as error:
            self.log(f"Error reproduciendo la ráfaga: {error}")

    def enviar_ack(self):
        """Envía la confirmación ACK pausando el receptor antes (si no,
        se autocaptura su propia confirmación) y lo reanuda después."""
        if self.idx_out_actual is None:
            return
        self.detener_flujo_recepcion()
        try:
            self.transmitir_texto("ACK:OK", self.idx_out_actual)
        finally:
            self.resume_signal.emit()

    # ---------- Voz con pulsación ----------
    def iniciar_captura_global_mouse(self):
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()

    def on_click(self, x, y, button, pressed):
        if button != mouse.Button.x1 or self.modo_cache not in ("Transmisión", "Comunicación completa"):
            return
        if pressed and not self.is_ptt_pressed:
            self.is_ptt_pressed = True
            if self.is_listening_stream:
                self.detener_flujo_recepcion()
            self.set_status("ESTADO: CAPTURANDO MICRO…", "#f87171")
            threading.Thread(target=self.comenzar_grabacion_ptt, daemon=True).start()
        elif not pressed and self.is_ptt_pressed:
            self.is_ptt_pressed = False
            self.set_status("ESTADO: PROCESANDO VOZ…", "#fbbf24")

    def _abrir_input_stream_compatible(self, callback):
        """Algunos dispositivos/drivers (sobre todo en Windows) rechazan
        una frecuencia de muestreo arbitraria (p.ej. 16000 Hz fijo) con
        'Invalid sample rate'. Probamos primero la frecuencia nativa del
        propio micrófono y, si aun así falla, unas cuantas habituales."""
        candidatos = []
        try:
            candidatos.append(int(sd.query_devices(self.idx_in_actual)["default_samplerate"]))
        except Exception:
            pass
        for extra in (48000, 44100, 16000):
            if extra not in candidatos:
                candidatos.append(extra)

        ultimo_error = None
        for fs in candidatos:
            try:
                stream = sd.InputStream(samplerate=fs, channels=1, device=self.idx_in_actual, callback=callback)
                return stream, fs
            except Exception as error:
                ultimo_error = error
        raise ultimo_error

    def comenzar_grabacion_ptt(self):
        if self.idx_in_actual is None or self.idx_out_actual is None:
            self.log("No hay dispositivos de entrada/salida seleccionados.")
            return
        captured = []
        try:
            stream, fs = self._abrir_input_stream_compatible(
                lambda data, *_: captured.append(data.copy()) if self.is_ptt_pressed else None
            )
            with stream:
                while self.is_ptt_pressed:
                    sd.sleep(40)
            if not captured:
                return
            stream = io.BytesIO()
            wav.write(stream, fs, (np.concatenate(captured) * 32767).astype(np.int16))
            stream.seek(0)
            with sr.AudioFile(stream) as source:
                text = self.recognizer.recognize_google(self.recognizer.record(source), language=self.IDIOMAS.get(self.idioma_cache, "es-ES"))
            self.log(f"Texto procesado: '{text}'")
            if self.validar_longitud_mensaje(text):
                self.add_history_threadsafe("Enviado", text)
                self.transmitir_texto(text, self.idx_out_actual)
        except sr.UnknownValueError:
            self.log("No se detectó una voz clara.")
        except Exception as error:
            self.log(f"Error en transmisión de voz: {error}")
        finally:
            self.set_status("ESTADO: Listo")
            if self.modo_cache == "Comunicación completa":
                self.resume_signal.emit()

    # ---------- Recepción ----------
    def actualizar_modo_operacion(self, mode):
        self.modo_cache = mode
        self.log(f"Cambiando a modo: [{mode}]")
        if mode == "Transmisión":
            self.detener_flujo_recepcion()
        else:
            self.iniciar_flujo_recepcion()

    def iniciar_flujo_recepcion(self):
        if self.is_listening_stream or self.idx_in_actual is None:
            return
        try:
            self.stream_receptor = sd.InputStream(samplerate=48000, channels=1, device=self.idx_in_actual, callback=self.callback_receptor, blocksize=1024)
            self.stream_receptor.start()
            self.is_listening_stream = True
            self.log("Receptor escuchando el espectro de audio…")
        except Exception as error:
            self.log(f"Error abriendo receptor: {error}")

    def detener_flujo_recepcion(self):
        if not self.is_listening_stream:
            return
        try:
            self.stream_receptor.stop()
            self.stream_receptor.close()
            self.is_listening_stream = False
            self.log("Receptor apagado.")
        except Exception as error:
            self.log(f"Error al apagar receptor: {error}")

    def callback_receptor(self, indata, frames, time_info, status):
        # Diezmado: solo 1 de cada 4 bloques actualiza el medidor visual,
        # para no repintar el gráfico ~47 veces por segundo sin necesidad.
        self._meter_counter += 1
        if self._meter_counter % 4 == 0:
            self.meter_signal.emit(float(min(1, np.max(np.abs(indata)))))

        with self.ggwave_lock:
            result = ggwave.decode(self.ggwave_instance, np.ascontiguousarray(indata[:, 0], dtype=np.float32).tobytes())
        if not result:
            return
        try:
            text = result.decode("utf-8")
            if text.startswith("ACK:"):
                self.log("Confirmación de recepción recibida.")
                self.add_history_threadsafe("Confirmado", "El otro equipo confirmó tu mensaje")
                return
            self.log(f"¡Ráfaga capturada!: {text}")
            self.add_history_threadsafe("Recibido", text)
            self.notification_signal.emit("Mensaje recibido", text)
            threading.Thread(target=self.ejecutar_tts, args=(text, self.tts_speed_cache), daemon=True).start()
            if self.confirmar_cache and self.idx_out_actual is not None:
                threading.Thread(target=self.enviar_ack, daemon=True).start()
        except Exception as error:
            self.log(f"Error decodificando señal: {error}")

    def ejecutar_tts(self, text, speed):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", speed)
            engine.say(text)
            engine.runAndWait()
        except Exception as error:
            self.log(f"Fallo en síntesis TTS: {error}")

    def closeEvent(self, event):
        if not self.closed:
            self.closed = True
            self.save_preferences()
            self.settings.setValue("geometry", self.saveGeometry())
            if self.mouse_listener:
                self.mouse_listener.stop()
            self.detener_flujo_recepcion()
            try:
                ggwave.free(self.ggwave_instance)
            except Exception:
                pass
            self.tray.hide()
        event.accept()


STYLESHEET = """
QWidget#root {
    background: qradialgradient(cx:0.15, cy:0.05, radius:1.25, fx:0.15, fy:0.05,
        stop:0 #2b2830, stop:0.38 #1a1822, stop:1 #0d0c12);
    color: #e8e8ef; font-family: 'Segoe UI Variable', 'Segoe UI'; font-size: 13px;
}
QTabWidget::pane { border: none; }
QTabBar::tab {
    background: transparent; color: #9a97a8; padding: 9px 18px; margin-right: 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600;
}
QTabBar::tab:selected { color: #ffffff; background: rgba(124, 92, 255, 60); }
QTabBar::tab:hover:!selected { color: #d6d3e0; }
QGroupBox {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(46,43,56,210), stop:1 rgba(24,22,32,225));
    border: 1px solid rgba(124,92,255,45); border-radius: 15px; margin-top: 14px;
    padding: 15px; font-weight: 600; color: #f4f4f5;
}
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 7px; color: #22d3ee; }
QLabel#title { color: #ffffff; font-size: 27px; font-weight: 700; letter-spacing: 0.5px; }
QLabel#subtitle, QLabel#muted { color: #9a97a8; }
QLabel#status { font-weight: 700; font-size: 14px; }
QPlainTextEdit, QComboBox, QTableWidget {
    background: rgba(10,9,15,165); color: #eeeeee; border: 1px solid rgba(255,255,255,28);
    border-radius: 10px; padding: 7px; selection-background-color: #7c5cff;
}
QPlainTextEdit:focus, QComboBox:focus { border: 1px solid #7c5cff; background: rgba(18,16,26,220); }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: none; border: solid #bcbcc0; border-width: 0 2px 2px 0; width: 6px; height: 6px; transform: rotate(45deg); }
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(70,66,84,225), stop:0.48 rgba(48,45,60,225), stop:1 rgba(30,28,38,230));
    color: #f8f8f8; border: 1px solid rgba(255,255,255,35); border-radius: 10px; padding: 10px 14px; font-weight: 600;
}
QPushButton:hover { border-color: rgba(124,92,255,150); }
QPushButton:pressed { padding-top: 12px; padding-bottom: 8px; }
QPushButton#accent {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b7bff, stop:0.5 #7c5cff, stop:1 #6540e0);
    color: #ffffff; border-color: rgba(255,255,255,60);
}
QPushButton#accent:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ab8eff, stop:0.5 #8c6cff, stop:1 #7350f0); }
QPushButton:disabled { color: #77748a; background: #2a2833; border-color: #3a3745; }

QFrame#modeSwitch { background: rgba(0,0,0,130); border: 1px solid rgba(124,92,255,60); border-radius: 11px; }
QPushButton#modeSeg {
    background: transparent; border: none; color: #b4b1c2; padding: 7px 14px;
    font-weight: 700; font-size: 12px; border-radius: 8px;
}
QPushButton#modeSeg:hover { color: #ffffff; }
QPushButton#modeSeg:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b7bff, stop:1 #6c4bd6);
    color: #ffffff;
}

QProgressBar { background: rgba(5,5,10,170); border: 1px solid rgba(255,255,255,20); border-radius: 5px; height: 10px; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff, stop:1 #22d3ee); border-radius: 5px; }
QHeaderView::section { background: rgba(124,92,255,45); color: #eeeeef; border: none; border-bottom: 1px solid rgba(255,255,255,30); padding: 7px; font-weight: 600; }
QTableWidget { gridline-color: rgba(255,255,255,15); alternate-background-color: rgba(255,255,255,6); }
QTableWidget::item:selected { background: rgba(124,92,255,90); color: #ffffff; }
QCheckBox { spacing: 8px; color: #dedee0; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #77737f; background: #201e28; border-radius: 5px; }
QCheckBox::indicator:checked { background: #9b7bff; border-color: #ffffff; }
QSlider::groove:horizontal { height: 6px; background: #201e28; border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c5cff, stop:1 #22d3ee); border-radius: 3px; }
QSlider::handle:horizontal { width: 17px; margin: -6px 0; background: #ffffff; border: 1px solid #7c5cff; border-radius: 8px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: rgba(124,92,255,120); border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: rgba(124,92,255,200); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


if __name__ == "__main__":
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    window = GibberLinkApp()
    window.show()
    app.exec()
