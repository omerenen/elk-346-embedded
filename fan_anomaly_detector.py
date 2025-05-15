#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fan_anomaly_detector.py

Fan anomaly detection system using:
 1) Reads data from raw ADC or time/amplitude CSV files
 2) Extracts time-domain and spectral features
 3) Trains and tests features with SVM
 4) Displays real-time visualizations through PyQt5 interface
 5) Processes serial data for real-time monitoring
 6) Sends Telegram notifications for persistent anomalies
"""


import os
import glob
import sys
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import time
import asyncio
from telegram import Bot
from telegram.error import TelegramError
from collections import deque
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QTabWidget, QFileDialog, 
                             QTextEdit, QLineEdit, QGroupBox, QGridLayout, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QStatusBar, QFrame, QSplitter)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import requests

# ---------------------------
# 1) GLOBAL PARAMETERS
# ---------------------------
FS = 8000           # Sampling frequency (Hz)
ADC_BITS = 12       # ADC resolution (bits)
VREF = 3.3          # ADC reference voltage (V)
WINDOW_SIZE = 1     # Window size for analysis in seconds
BUFFER_SIZE = 10    # Number of windows to store for display

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7705153823:AAGcRQdvJ4SMHlKf_3h8dT6sfZjUzNT2ZJk"  # Replace with your bot token
TELEGRAM_CHAT_ID = "874605924"      # Replace with your chat ID
ANOMALY_THRESHOLD_TIME = 1.0           # Time in seconds before sending alert

# Frequency bands we want to analyze (Hz)
BANDS = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000)
]

# Default colors
COLOR_NORMAL = '#4CAF50'    # Green
COLOR_ANOMALY = '#F44336'   # Red
COLOR_SIGNAL = '#2196F3'    # Blue
COLOR_SPECTRUM = '#FF9800'  # Orange
COLOR_BG_DARK = '#263238'   # Dark blue-gray
COLOR_TEXT_LIGHT = '#ECEFF1' # Light gray
COLOR_PANEL = '#37474F'     # Darker blue-gray

# ---------------------------
# 2) FEATURE EXTRACTION FUNCTIONS
# ---------------------------

def extract_band_energies(signal: np.ndarray, fs: int, bands=BANDS) -> np.ndarray:
    """
    Calculates total energy in specific frequency bands.
    signal: normalized time-domain signal array
    fs: sampling frequency
    returns: vector containing a single number (energy) for each band
    """
    N = len(signal)
    # Hanning window + FFT
    windowed = signal * np.hanning(N)
    Y = np.fft.rfft(windowed)
    P = np.abs(Y)**2
    freqs = np.fft.rfftfreq(N, 1/fs)

    feats = []
    for fmin, fmax in bands:
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        feats.append(P[idx].sum())  # total energy in band
    return np.array(feats)


def extract_features(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    For a single window (e.g., 1s), collects features in a single vector:
     - RMS energy
     - Zero-Crossing Rate
     - Spectral centroid, bandwidth, rolloff
     - Band energies
    """
    N = len(signal)

    # ---- Time-domain features ----
    rms = np.sqrt(np.mean(signal**2))                           # signal power
    zcr = np.mean(np.abs(np.diff(np.sign(signal))) > 0)         # zero crossing rate
    peak = np.max(np.abs(signal))                               # peak amplitude
    crest = peak / (rms + 1e-10)                               # crest factor

    # ---- Spectral preparation ----
    windowed = signal * np.hanning(N)
    Y = np.fft.rfft(windowed)
    P = np.abs(Y)**2
    freqs = np.fft.rfftfreq(N, 1/fs)

    # Normalize power spectrum
    P = P / (np.sum(P) + 1e-10)

    # 1) Spectral centroid
    centroid = np.sum(freqs * P)
        
    # 2) Spectral bandwidth (standard deviation)
    bandwidth = np.sqrt(np.sum(((freqs-centroid)**2) * P))
        
    # 3) Spectral rolloff (lower limit of 85% energy)
    cumulative = np.cumsum(P)
    rolloff_idx = np.searchsorted(cumulative, 0.85)
    rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]

    # 4) Spectral flatness
    geometric_mean = np.exp(np.mean(np.log(P + 1e-10)))
    arithmetic_mean = np.mean(P)
    flatness = geometric_mean / (arithmetic_mean + 1e-10)

    # ---- Band energies ----
    band_feats = extract_band_energies(signal, fs)
    
    # Normalize band energies
    total_energy = np.sum(band_feats)
    if total_energy > 0:
        band_feats = band_feats / total_energy

    # ---- Combine all ----
    return np.hstack([rms, zcr, peak, crest, centroid, bandwidth, rolloff, flatness, band_feats])


def get_fft_data(signal: np.ndarray, fs: int):
    """Calculate FFT for visualization"""
    N = len(signal)
    window = np.hanning(N)
    windowed = signal * window
    
    Y = np.fft.rfft(windowed)
    P = np.abs(Y)**2
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    return freqs, P


def compute_spectrogram(signal: np.ndarray, fs: int):
    """Compute spectrogram for visualization"""
    f, t, Sxx = spectrogram(signal, fs, nperseg=256, noverlap=128)
    return f, t, 10 * np.log10(Sxx + 1e-10)  # in dB


# ---------------------------
# 3) DATASET CREATION
# ---------------------------

def load_dataset(normal_dir: str, anomaly_dir: str):
    """
    normal_dir: directory with normal operation CSVs, like 'data/normal'
    anomaly_dir: directory with anomaly CSVs, like 'data/anomaly'
    returns: X (num_examples x num_features), y (0=normal, 1=anomaly)
    """
    X, y = [], []

    # For each folder, assign label: normal->0, anomaly->1
    for label, folder in enumerate([normal_dir, anomaly_dir]):
        pattern = os.path.join(folder, '*.csv')
        for fn in glob.glob(pattern):
            try:
                df = pd.read_csv(fn)

                # Get signal: use amplitude column if available, otherwise ADC->volt->normalize
                if 'amplitude' in df.columns:
                    sig = df['amplitude'].values
                else:
                    # Assuming ADC values are in the second column
                    counts = df.iloc[:, 1].values
                    volt = counts / (2**ADC_BITS - 1) * VREF
                    sig = (volt - VREF/2) / (VREF/2)

                # If signal length is different from fs, cut/resample
                # Add this if needed.

                feats = extract_features(sig, FS)
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"Error processing {fn}: {e}")

    # List → NumPy array
    if X:  # Check if X is not empty
        X = np.vstack(X)
        y = np.array(y)
    else:
        # Create empty arrays with correct dimensions if no data
        X = np.empty((0, 9))  # 9 features (adjust if different)
        y = np.array([])
        
    return X, y


# ---------------------------
# 4) SERIAL DATA ACQUISITION
# ---------------------------

class SerialWorker(QThread):
    data_received = pyqtSignal(np.ndarray)
    raw_data_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, port=None, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.buffer = []
        self.raw_data_buffer = []
        self.samples_to_collect = FS * WINDOW_SIZE
        self.last_emit_time = 0
        
        # Sentetik veri için değişkenler
        self.is_synthetic = False
        self.synthetic_data = None
        self.current_index = 0
        self.is_anomaly = False
        
    def set_port(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        
    def set_synthetic_mode(self, is_anomaly=False):
        """Sentetik veri modunu ayarla"""
        self.is_synthetic = True
        self.is_anomaly = is_anomaly
        self.generate_synthetic_data()
        
    def generate_synthetic_data(self):
        """Fan sesi için sentetik veri üret"""
        try:
            # Veri dosyasını yükle
            data_type = "anomaly" if self.is_anomaly else "normal"
            sample_file = f"data/{data_type}/{data_type}_fan_1.csv"
            
            if not os.path.exists(sample_file):
                self.error_occurred.emit(f"Sample data not found. Please run generate_fan_data.py first.")
                return
                
            df = pd.read_csv(sample_file)
            self.synthetic_data = df['amplitude'].values
            self.current_index = 0
        except Exception as e:
            self.error_occurred.emit(f"Error loading synthetic data: {str(e)}")
        
    def stop(self):
        """Stop the serial worker and clean up"""
        self.running = False
        self.wait()
        if self.serial and self.serial.is_open:
            self.serial.close()
        
        # Bufferları temizle
        self.buffer = []
        self.raw_data_buffer = []
        self.synthetic_data = None
        self.current_index = 0
        self.is_synthetic = False
        
        print("Serial worker stopped and buffers cleared")
            
    def run(self):
        if self.is_synthetic:
            self.run_synthetic()
        else:
            self.run_serial()
            
    def run_synthetic(self):
        """Sentetik veri modunda çalış"""
        self.running = True
        
        while self.running and self.synthetic_data is not None:
            try:
                # Her seferinde 1000 örnek gönder (bir pencere büyüklüğü)
                chunk_size = min(1000, len(self.synthetic_data))
                end_idx = self.current_index + chunk_size
                
                if end_idx >= len(self.synthetic_data):
                    # Başa dön
                    self.current_index = 0
                    end_idx = chunk_size
                
                # Veriyi al ve gönder
                chunk = self.synthetic_data[self.current_index:end_idx]
                self.current_index = end_idx
                
                # ADC değerlerini simüle et (12-bit ADC için 0-4095 arası)
                adc_values = (chunk * 2047.5 + 2047.5).astype(int)
                
                # Ham veri sinyali gönder
                raw_str = " ".join(str(v) for v in adc_values[:5])  # İlk 5 değeri göster
                self.raw_data_received.emit(raw_str)
                
                # İşlenmiş veriyi gönder
                self.data_received.emit(chunk)
                
                # 100ms bekle
                time.sleep(0.1)
                
            except Exception as e:
                self.error_occurred.emit(f"Synthetic data error: {str(e)}")
                break
                
    def run_serial(self):
        """Normal serial port modunda çalış"""
        if not self.port:
            self.error_occurred.emit("No port selected")
            return
            
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.buffer = []
            self.raw_data_buffer = []
            
            while self.running:
                if self.serial.in_waiting > 0:
                    try:
                        raw = self.serial.readline()
                        if not raw:
                            continue
                            
                        raw_str = " ".join(str(b) for b in raw)
                        self.raw_data_received.emit(raw_str)
                            
                        try:
                            numbers = [int(b) for b in raw_str.split()]
                            values = []
                            for i in range(0, len(numbers), 2):
                                if i < len(numbers):
                                    values.append(numbers[i])
                            
                            if values:
                                for value in values:
                                    volt = value / (2**ADC_BITS - 1) * VREF
                                    normalized = (volt - VREF/2) / (VREF/2)
                                    self.buffer.append(normalized)
                                    self.raw_data_buffer.append(value)
                                
                                if len(self.buffer) >= self.samples_to_collect:
                                    data_array = np.array(self.buffer)
                                    self.data_received.emit(data_array)
                                    self.buffer = self.buffer[len(self.buffer)//2:]
                                    
                        except Exception as e:
                            print(f"Data processing error: {e}")
                    except Exception as e:
                        print(f"Read error: {e}")
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.error_occurred.emit(f"Serial error: {str(e)}")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()


# ---------------------------
# WORKER THREADS
# ---------------------------

class DataProcessWorker(QThread):
    """Veri işleme için ayrı thread"""
    processing_complete = pyqtSignal(np.ndarray, object)  # Signal: (signal_data, features)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, signal_data):
        super().__init__()
        self.signal_data = signal_data
        
    def run(self):
        try:
            # Extract features
            features = extract_features(self.signal_data, FS)
            
            # Emit result
            self.processing_complete.emit(self.signal_data, features)
        except Exception as e:
            self.error_occurred.emit(str(e))
            import traceback
            print(f"Data processing error: {traceback.format_exc()}")


class PlotUpdateWorker(QThread):
    """Grafik güncelleme için ayrı thread"""
    plot_data_ready = pyqtSignal(object, object, object)  # Signal: (time_data, freq_data, spec_data)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, signal_data):
        super().__init__()
        self.signal_data = signal_data
        
    def run(self):
        try:
            # Prepare time domain data
            time_axis = np.linspace(0, len(self.signal_data)/FS, len(self.signal_data))
            time_data = (time_axis, self.signal_data)
            
            # Prepare frequency domain data
            freqs, power = get_fft_data(self.signal_data, FS)
            freq_data = (freqs, power)
            
            # Prepare spectrogram data
            f, t, Sxx = compute_spectrogram(self.signal_data, FS)
            spec_data = (f, t, Sxx)
            
            # Emit results
            self.plot_data_ready.emit(time_data, freq_data, spec_data)
        except Exception as e:
            self.error_occurred.emit(str(e))
            import traceback
            print(f"Plot update error: {traceback.format_exc()}")


# ---------------------------
# 5) GUI COMPONENTS
# ---------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.set_facecolor(COLOR_BG_DARK)
        self.axes.set_facecolor(COLOR_BG_DARK)
        self.axes.tick_params(colors=COLOR_TEXT_LIGHT)
        self.axes.spines['bottom'].set_color(COLOR_TEXT_LIGHT)
        self.axes.spines['top'].set_color(COLOR_TEXT_LIGHT)
        self.axes.spines['left'].set_color(COLOR_TEXT_LIGHT)
        self.axes.spines['right'].set_color(COLOR_TEXT_LIGHT)
        self.axes.xaxis.label.set_color(COLOR_TEXT_LIGHT)
        self.axes.yaxis.label.set_color(COLOR_TEXT_LIGHT)
        self.axes.title.set_color(COLOR_TEXT_LIGHT)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup main window
        self.setWindowTitle("Fan Anomaly Detection System")
        self.setMinimumSize(1200, 800)
        
        # Set up dark theme
        self.setup_dark_theme()
        
        # Initialize variables
        self.model = None
        self.scaler = None
        self.signal_buffer = deque(maxlen=BUFFER_SIZE)
        self.raw_data_buffer = deque(maxlen=100)  # Son 100 ham veriyi sakla
        self.feature_names = ['RMS', 'ZCR', 'Peak', 'Crest', 'Centroid', 'Bandwidth', 'Rolloff', 'Flatness'] + [f'Band {i+1}' for i in range(len(BANDS))]
        
        # Anomaly tracking
        self.anomaly_start_time = None
        self.last_anomaly_notification_time = 0
        self.notification_cooldown = 60  # Minimum seconds between notifications
        
        # Sample data playback variables
        self.current_sample_index = 0
        self.sample_data = []
        self.sample_timer = QTimer()
        self.sample_timer.timeout.connect(self.update_sample_display)
        
        # Thread yönetimi
        self.data_process_worker = None
        self.plot_update_worker = None
        
        # Serial worker setup
        self.serial_worker = SerialWorker()
        self.serial_worker.data_received.connect(self.process_new_data)
        self.serial_worker.raw_data_received.connect(self.process_raw_data)
        self.serial_worker.error_occurred.connect(self.show_error)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 10))
        
        # Tab 1: Real-time monitoring
        self.monitoring_tab = QWidget()
        self.setup_monitoring_tab()
        self.tabs.addTab(self.monitoring_tab, "Real-time Monitoring")
        
        # Tab 2: Training and evaluation
        self.training_tab = QWidget()
        self.setup_training_tab()
        self.tabs.addTab(self.training_tab, "Training & Evaluation")
        
        # Tab 3: Settings
        self.settings_tab = QWidget()
        self.setup_settings_tab()
        self.tabs.addTab(self.settings_tab, "Settings")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Initialize plots
        self.clear_plots()
        
        # Telegram bot initialization - moved here after UI setup
        try:
            self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
            self.add_log("Telegram bot initialized successfully")
        except Exception as e:
            self.bot = None
            self.add_log(f"Failed to initialize Telegram bot: {str(e)}", error=True)
        
    def setup_dark_theme(self):
        """Setup dark theme for the application"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(COLOR_BG_DARK))
        dark_palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT_LIGHT))
        dark_palette.setColor(QPalette.Base, QColor(COLOR_PANEL))
        dark_palette.setColor(QPalette.AlternateBase, QColor(COLOR_BG_DARK))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(COLOR_TEXT_LIGHT))
        dark_palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT_LIGHT))
        dark_palette.setColor(QPalette.Text, QColor(COLOR_TEXT_LIGHT))
        dark_palette.setColor(QPalette.Button, QColor(COLOR_PANEL))
        dark_palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT_LIGHT))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(COLOR_TEXT_LIGHT))
        
        QApplication.setPalette(dark_palette)
        QApplication.setStyle("Fusion")
        
    def setup_monitoring_tab(self):
        """Setup real-time monitoring tab"""
        layout = QVBoxLayout(self.monitoring_tab)
        
        # Connection controls
        connection_group = QGroupBox("Connection & Data Controls")
        connection_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        connection_layout = QHBoxLayout(connection_group)
        
        # Serial port controls
        port_group = QWidget()
        port_layout = QHBoxLayout(port_group)
        
        self.port_combo = QComboBox()
        self.refresh_ports_btn = QPushButton("Refresh Ports")
        self.refresh_ports_btn.clicked.connect(self.refresh_serial_ports)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.refresh_ports_btn)
        port_layout.addWidget(self.connect_btn)
        
        # Initialize port list
        self.refresh_serial_ports()
        
        # Synthetic data controls
        synthetic_group = QWidget()
        synthetic_layout = QHBoxLayout(synthetic_group)
        
        self.normal_data_btn = QPushButton("Normal Fan")
        self.normal_data_btn.clicked.connect(lambda: self.start_synthetic_data(False))
        self.anomaly_data_btn = QPushButton("Anomaly Fan")
        self.anomaly_data_btn.clicked.connect(lambda: self.start_synthetic_data(True))
        self.stop_data_btn = QPushButton("Stop")
        self.stop_data_btn.clicked.connect(self.stop_data)
        self.stop_data_btn.setEnabled(False)
        
        synthetic_layout.addWidget(QLabel("Simulate:"))
        synthetic_layout.addWidget(self.normal_data_btn)
        synthetic_layout.addWidget(self.anomaly_data_btn)
        synthetic_layout.addWidget(self.stop_data_btn)
        
        # Add both groups to connection layout
        connection_layout.addWidget(port_group)
        connection_layout.addWidget(QLabel("|"))  # Separator
        connection_layout.addWidget(synthetic_group)
        connection_layout.addStretch()
        
        # Raw data and logs display
        data_log_splitter = QSplitter(Qt.Horizontal)
        
        # Raw data display
        raw_data_group = QGroupBox("Raw Serial Data")
        raw_data_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        raw_data_layout = QVBoxLayout(raw_data_group)
        
        self.raw_data_text = QTextEdit()
        self.raw_data_text.setReadOnly(True)
        self.raw_data_text.setMaximumHeight(100)
        self.raw_data_text.setFont(QFont("Courier New", 9))
        self.raw_data_text.setStyleSheet(f"""
            QTextEdit {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 5px;
            }}
        """)
        raw_data_layout.addWidget(self.raw_data_text)
        
        # Log display
        log_group = QGroupBox("System Logs")
        log_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 5px;
            }}
        """)
        log_layout.addWidget(self.log_text)
        
        # Add groups to splitter
        data_log_splitter.addWidget(raw_data_group)
        data_log_splitter.addWidget(log_group)
        data_log_splitter.setSizes([int(data_log_splitter.width() * 0.5)] * 2)  # Equal sizes
        
        # ADC Value display
        self.adc_value_label = QLabel("")
        self.adc_value_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.adc_value_label.setAlignment(Qt.AlignCenter)
        
        # Visualization area
        self.main_splitter = QSplitter(Qt.Vertical)
        self.top_splitter = QSplitter(Qt.Horizontal)
        
        # Time domain plot
        time_group = QGroupBox("Time Domain Signal")
        time_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        time_layout = QVBoxLayout(time_group)
        self.time_canvas = MplCanvas(width=6, height=3)
        time_layout.addWidget(self.time_canvas)
        
        # Frequency domain plot
        freq_group = QGroupBox("Frequency Domain")
        freq_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        freq_layout = QVBoxLayout(freq_group)
        self.freq_canvas = MplCanvas(width=6, height=3)
        freq_layout.addWidget(self.freq_canvas)
        
        self.top_splitter.addWidget(time_group)
        self.top_splitter.addWidget(freq_group)
        
        # Spectrogram
        spec_group = QGroupBox("Spectrogram")
        spec_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        spec_layout = QVBoxLayout(spec_group)
        self.spec_canvas = MplCanvas(width=12, height=3)
        spec_layout.addWidget(self.spec_canvas)
        
        # Detection result display
        result_group = QGroupBox("Detection Result")
        result_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        result_layout = QHBoxLayout(result_group)
        
        self.result_label = QLabel("No Data")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        result_layout.addWidget(self.result_label)
        
        # Add all to main splitter
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(spec_group)
        self.main_splitter.addWidget(result_group)
        
        # Add to main layout
        layout.addWidget(connection_group)
        layout.addWidget(data_log_splitter)
        layout.addWidget(self.adc_value_label)
        layout.addWidget(self.main_splitter)
        
    def setup_training_tab(self):
        """Setup training and evaluation tab"""
        layout = QVBoxLayout(self.training_tab)
        layout.setSpacing(5)  # Reduce spacing between elements
        
        # Data selection area
        data_group = QGroupBox("Dataset Selection")
        data_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        data_group.setStyleSheet(f"""
            QGroupBox {{ 
                color: {COLOR_TEXT_LIGHT}; 
                border: 1px solid {COLOR_TEXT_LIGHT}; 
                border-radius: 5px; 
                margin-top: 10px; 
                padding: 5px; 
            }}
        """)
        data_layout = QGridLayout(data_group)
        data_layout.setSpacing(5)  # Reduce spacing
        data_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        self.normal_path = QLineEdit()
        self.normal_path.setPlaceholderText("Path to folder with normal fan data")
        self.normal_path.setStyleSheet(f"""
            QLineEdit {{ 
                color: {COLOR_TEXT_LIGHT}; 
                background-color: {COLOR_PANEL}; 
                border: 1px solid {COLOR_TEXT_LIGHT}; 
                border-radius: 3px; 
                padding: 3px; 
                height: 20px; 
            }}
        """)
        self.normal_browse = QPushButton("Browse...")
        self.normal_browse.setStyleSheet(f"""
            QPushButton {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 3px 10px;
                height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_SIGNAL};
            }}
        """)
        self.normal_browse.clicked.connect(lambda: self.browse_folder(self.normal_path))
        
        self.anomaly_path = QLineEdit()
        self.anomaly_path.setPlaceholderText("Path to folder with anomaly fan data")
        self.anomaly_path.setStyleSheet(f"""
            QLineEdit {{ 
                color: {COLOR_TEXT_LIGHT}; 
                background-color: {COLOR_PANEL}; 
                border: 1px solid {COLOR_TEXT_LIGHT}; 
                border-radius: 3px; 
                padding: 3px; 
                height: 20px; 
            }}
        """)
        self.anomaly_browse = QPushButton("Browse...")
        self.anomaly_browse.setStyleSheet(f"""
            QPushButton {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 3px 10px;
                height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_SIGNAL};
            }}
        """)
        self.anomaly_browse.clicked.connect(lambda: self.browse_folder(self.anomaly_path))
        
        path_label_style = f"QLabel {{ color: {COLOR_TEXT_LIGHT}; font-weight: bold; }}"
        normal_label = QLabel("Normal:")
        normal_label.setStyleSheet(path_label_style)
        anomaly_label = QLabel("Anomaly:")
        anomaly_label.setStyleSheet(path_label_style)
        
        data_layout.addWidget(normal_label, 0, 0)
        data_layout.addWidget(self.normal_path, 0, 1)
        data_layout.addWidget(self.normal_browse, 0, 2)
        data_layout.addWidget(anomaly_label, 1, 0)
        data_layout.addWidget(self.anomaly_path, 1, 1)
        data_layout.addWidget(self.anomaly_browse, 1, 2)
        
        # Training controls - daha kompakt tasarım
        train_group = QGroupBox("Training Controls")
        train_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        train_group.setStyleSheet(f"""
            QGroupBox {{ 
                color: {COLOR_TEXT_LIGHT}; 
                border: 1px solid {COLOR_TEXT_LIGHT}; 
                border-radius: 5px; 
                margin-top: 5px; 
                padding: 5px; 
            }}
        """)
        train_layout = QHBoxLayout(train_group)  # Changed to QHBoxLayout for horizontal arrangement
        train_layout.setSpacing(10)
        train_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Test size kontrolü
        test_size_label = QLabel("Test Size:")
        test_size_label.setStyleSheet(path_label_style)
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setFixedWidth(70)  # Set fixed width
        self.test_size_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 2px;
                height: 20px;
            }}
        """)
        
        # C parametresi kontrolü
        c_param_label = QLabel("C Parameter:")
        c_param_label.setStyleSheet(path_label_style)
        self.c_param = QDoubleSpinBox()
        self.c_param.setRange(0.1, 10.0)
        self.c_param.setValue(1.0)
        self.c_param.setSingleStep(0.1)
        self.c_param.setFixedWidth(70)  # Set fixed width
        self.c_param.setStyleSheet(f"""
            QDoubleSpinBox {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 2px;
                height: 20px;
            }}
        """)
        
        # Train butonu
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setStyleSheet(f"""
            QPushButton {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_NORMAL};
                border: none;
                border-radius: 3px;
                padding: 3px 15px;
                height: 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2E7D32;
            }}
        """)
        self.train_btn.clicked.connect(self.train_model)
        
        # Add controls to train layout horizontally
        train_layout.addWidget(test_size_label)
        train_layout.addWidget(self.test_size_spin)
        train_layout.addWidget(c_param_label)
        train_layout.addWidget(self.c_param)
        train_layout.addWidget(self.train_btn)
        train_layout.addStretch()
        
        # Results display
        result_group = QGroupBox("Training Results")
        result_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        result_group.setStyleSheet(f"""
            QGroupBox {{ 
                color: {COLOR_TEXT_LIGHT}; 
                border: 1px solid {COLOR_TEXT_LIGHT}; 
                border-radius: 5px; 
                margin-top: 5px; 
                padding: 5px; 
            }}
        """)
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet(f"""
            QTextEdit {{
                color: {COLOR_TEXT_LIGHT};
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_TEXT_LIGHT};
                border-radius: 3px;
                padding: 5px;
                font-family: 'Courier New';
                font-size: 10pt;
            }}
        """)
        result_layout.addWidget(self.result_text)
        
        # Features visualization
        feature_group = QGroupBox("Feature Importance")
        feature_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        feature_group.setStyleSheet(f"""
            QGroupBox {{ 
                color: {COLOR_TEXT_LIGHT}; 
                border: 1px solid {COLOR_TEXT_LIGHT}; 
                border-radius: 5px; 
                margin-top: 5px; 
                padding: 5px; 
            }}
        """)
        feature_layout = QVBoxLayout(feature_group)
        feature_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        self.features_canvas = MplCanvas(width=12, height=4)
        feature_layout.addWidget(self.features_canvas)
        
        # Add to main layout with minimal spacing
        layout.addWidget(data_group)
        layout.addWidget(train_group)
        
        # Split results and features
        results_features = QSplitter(Qt.Horizontal)
        results_features.addWidget(result_group)
        results_features.addWidget(feature_group)
        layout.addWidget(results_features)
        
        # Set stretch factors
        layout.setStretch(0, 0)  # Dataset Selection - minimum height
        layout.setStretch(1, 0)  # Training Controls - minimum height
        layout.setStretch(2, 1)  # Results and Features - takes remaining space

    def setup_settings_tab(self):
        """Setup settings tab"""
        layout = QVBoxLayout(self.settings_tab)
        
        # General settings
        general_group = QGroupBox("Analysis Settings")
        general_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        general_layout = QGridLayout(general_group)
        
        self.sampling_rate = QSpinBox()
        self.sampling_rate.setRange(1000, 48000)
        self.sampling_rate.setValue(FS)
        self.sampling_rate.setSingleStep(1000)
        
        self.adc_bits = QSpinBox()
        self.adc_bits.setRange(8, 24)
        self.adc_bits.setValue(ADC_BITS)
        
        self.ref_voltage = QDoubleSpinBox()
        self.ref_voltage.setRange(1.0, 5.0)
        self.ref_voltage.setValue(VREF)
        self.ref_voltage.setSingleStep(0.1)
        
        general_layout.addWidget(QLabel("Sampling Rate (Hz):"), 0, 0)
        general_layout.addWidget(self.sampling_rate, 0, 1)
        general_layout.addWidget(QLabel("ADC Resolution (bits):"), 1, 0)
        general_layout.addWidget(self.adc_bits, 1, 1)
        general_layout.addWidget(QLabel("Reference Voltage (V):"), 2, 0)
        general_layout.addWidget(self.ref_voltage, 2, 1)
        
        # Serial settings
        serial_group = QGroupBox("Serial Settings")
        serial_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        serial_layout = QGridLayout(serial_group)
        
        self.baudrate = QComboBox()
        baudrates = ['9600', '19200', '38400', '57600', '115200', '230400', '460800', '921600']
        self.baudrate.addItems(baudrates)
        self.baudrate.setCurrentText('115200')
        
        serial_layout.addWidget(QLabel("Baud Rate:"), 0, 0)
        serial_layout.addWidget(self.baudrate, 0, 1)
        
        # Display settings
        display_group = QGroupBox("Display Settings")
        display_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        display_layout = QGridLayout(display_group)
        
        self.window_size = QDoubleSpinBox()
        self.window_size.setRange(0.1, 5.0)
        self.window_size.setValue(WINDOW_SIZE)
        self.window_size.setSingleStep(0.1)
        
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(1, 100)
        self.buffer_size.setValue(BUFFER_SIZE)
        
        self.show_spectrogram = QCheckBox("Show Spectrogram")
        self.show_spectrogram.setChecked(True)
        
        display_layout.addWidget(QLabel("Window Size (s):"), 0, 0)
        display_layout.addWidget(self.window_size, 0, 1)
        display_layout.addWidget(QLabel("Buffer Size:"), 1, 0)
        display_layout.addWidget(self.buffer_size, 1, 1)
        display_layout.addWidget(self.show_spectrogram, 2, 0, 1, 2)
        
        # Save button
        self.save_settings_btn = QPushButton("Apply Settings")
        self.save_settings_btn.clicked.connect(self.apply_settings)
        
        # Add to main layout
        layout.addWidget(general_group)
        layout.addWidget(serial_group)
        layout.addWidget(display_group)
        layout.addWidget(self.save_settings_btn)
        layout.addStretch()

    def refresh_serial_ports(self):
        """Refresh the list of available serial ports"""
        try:
            self.port_combo.clear()
            ports = [port.device for port in serial.tools.list_ports.comports()]
            
            if ports:
                self.port_combo.addItems(ports)
                self.statusBar().showMessage(f"Found {len(ports)} serial ports", 3000)
            else:
                self.port_combo.addItem("No ports available")
                self.statusBar().showMessage("No serial ports found", 3000)
                
            # Port listesi değiştiğinde Connect butonunun durumunu güncelle
            self.connect_btn.setEnabled(len(ports) > 0)
            
        except Exception as e:
            self.show_error(f"Error refreshing ports: {str(e)}")
            self.port_combo.addItem("Error listing ports")
            self.connect_btn.setEnabled(False)
        
    def browse_folder(self, line_edit):
        """Open folder browser dialog"""
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)
    
    def toggle_connection(self):
        """Toggle serial connection on/off"""
        if self.serial_worker.running:
            self.stop_data()
        else:
            port = self.port_combo.currentText()
            baudrate = int(self.baudrate.currentText())
            
            if port and port != "No ports available":
                self.serial_worker.set_port(port, baudrate)
                self.serial_worker.start()
                self.connect_btn.setText("Disconnect")
                self.add_log(f"Connected to {port} at {baudrate} baud")
                self.statusBar().showMessage(f"Connected to {port} at {baudrate} baud")
                self.result_label.setText("Waiting for data...")
                self.result_label.setStyleSheet(f"color: {COLOR_SIGNAL}; font-weight: bold;")
                
                # Disable synthetic data buttons when using real serial
                self.normal_data_btn.setEnabled(False)
                self.anomaly_data_btn.setEnabled(False)
                self.stop_data_btn.setEnabled(True)
            else:
                self.show_error("No valid port selected")
                
    def start_synthetic_data(self, is_anomaly):
        """Start synthetic data playback"""
        try:
            # Stop any existing data stream
            self.stop_data()
            
            # Configure and start synthetic data worker
            self.serial_worker.set_synthetic_mode(is_anomaly)
            self.serial_worker.start()
            
            # Update UI
            data_type = "Anomaly" if is_anomaly else "Normal"
            self.statusBar().showMessage(f"Playing synthetic {data_type.lower()} fan data")
            self.result_label.setText("Waiting for data...")
            self.result_label.setStyleSheet(f"color: {COLOR_SIGNAL}; font-weight: bold;")
            
            # Update button states
            self.normal_data_btn.setEnabled(False)
            self.anomaly_data_btn.setEnabled(False)
            self.stop_data_btn.setEnabled(True)
            self.connect_btn.setEnabled(False)
            
        except Exception as e:
            self.show_error(f"Error starting synthetic data: {str(e)}")
            
    def stop_data(self):
        """Stop all data streams"""
        # Stop the worker
        self.serial_worker.stop()
        
        # Reset UI
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.normal_data_btn.setEnabled(True)
        self.anomaly_data_btn.setEnabled(True)
        self.stop_data_btn.setEnabled(False)
        
        self.statusBar().showMessage("Data stream stopped")
        self.result_label.setText("No Data")
        self.result_label.setStyleSheet(f"color: {COLOR_TEXT_LIGHT}; font-weight: bold;")
        self.adc_value_label.setText("")
        
        # Clear plots
        self.clear_plots()
        
        # Clear buffers
        self.raw_data_text.clear()
        self.signal_buffer.clear()
        self.raw_data_buffer.clear()

    def apply_settings(self):
        """Apply new settings"""
        global FS, ADC_BITS, VREF, WINDOW_SIZE, BUFFER_SIZE
        
        FS = self.sampling_rate.value()
        ADC_BITS = self.adc_bits.value()
        VREF = self.ref_voltage.value()
        WINDOW_SIZE = self.window_size.value()
        BUFFER_SIZE = self.buffer_size.value()
        
        # Update serial worker samples
        self.serial_worker.samples_to_collect = int(FS * WINDOW_SIZE)
        
        # Update buffer
        new_buffer = deque(maxlen=BUFFER_SIZE)
        for item in self.signal_buffer:
            if len(new_buffer) < BUFFER_SIZE:
                new_buffer.append(item)
        self.signal_buffer = new_buffer
        
        self.statusBar().showMessage("Settings applied", 3000)
    
    def show_error(self, message):
        """Show error message in status bar and log"""
        self.statusBar().showMessage(f"Error: {message}", 5000)
        self.add_log(f"ERROR: {message}", error=True)
        print(f"Error: {message}")  # Also print to console for debugging

    def add_log(self, message, error=False, normal=False):
        """Add message to log display"""
        timestamp = time.strftime("%H:%M:%S")
        if error:
            color = COLOR_ANOMALY
        elif normal:
            color = COLOR_NORMAL
        else:
            color = COLOR_TEXT_LIGHT
            
        formatted_message = f"[{timestamp}] {message}"
        
        # Add new message at the top
        self.log_text.append(f'<span style="color: {color};">{formatted_message}</span>')
        
        # Scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def process_new_data(self, signal):
        """Process new data received from serial port or synthetic data"""
        try:
            # Sinyal boyutunu kontrol et
            if len(signal) < 10:  # Çok küçük sinyalleri işleme
                print(f"Signal too small: {len(signal)} samples")
                return
                
            # Add to buffer
            self.signal_buffer.append(signal)
            
            # Update status to show data is being received
            self.statusBar().showMessage(f"Processing signal data... ({len(signal)} samples)", 1000)
            
            # Önceki thread'in çalışması bitmişse temizle
            if self.data_process_worker is not None and self.data_process_worker.isFinished():
                self.data_process_worker = None
                
            # Veri işleme işini ayrı thread'e taşı
            if self.data_process_worker is None:
                self.data_process_worker = DataProcessWorker(signal)
                self.data_process_worker.processing_complete.connect(self.on_processing_complete)
                self.data_process_worker.error_occurred.connect(self.show_error)
                self.data_process_worker.start()
                
            # Grafik güncelleme işini ayrı thread'e taşı
            if self.plot_update_worker is None or self.plot_update_worker.isFinished():
                self.plot_update_worker = PlotUpdateWorker(signal)
                self.plot_update_worker.plot_data_ready.connect(self.update_plots_from_thread)
                self.plot_update_worker.error_occurred.connect(self.show_error)
                self.plot_update_worker.start()
                
        except Exception as e:
            self.show_error(f"Error processing data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def send_telegram_notification(self, message):
        """Telegram mesajını gönder"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                self.add_log("✅ Telegram notification sent successfully", normal=True)
                return True
            else:
                self.add_log(f"❌ Failed to send notification. Status code: {response.status_code}", error=True)
                return False
                
        except Exception as e:
            self.add_log(f"❌ Error sending notification: {str(e)}", error=True)
            return False

    def on_processing_complete(self, signal, features):
        """Veri işleme thread'i tamamlandığında çağrılır"""
        try:
            # Predict if model exists
            if self.model is not None and self.scaler is not None:
                # Scale features
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Get prediction and probability
                prediction = self.model.predict(features_scaled)[0]
                probability = self.model.predict_proba(features_scaled)[0][prediction]
                
                current_time = time.time()
                
                # Update result display
                if prediction == 0:  # Normal
                    result_text = f"NORMAL ({probability:.2f})"
                    self.result_label.setText(result_text)
                    self.result_label.setStyleSheet(f"color: {COLOR_NORMAL}; font-weight: bold;")
                    self.add_log(f"Detection Result: {result_text}", normal=True)
                    
                    # Reset anomaly tracking
                    self.anomaly_start_time = None
                    
                else:  # Anomaly
                    result_text = f"ANOMALY DETECTED ({probability:.2f})"
                    self.result_label.setText(result_text)
                    self.result_label.setStyleSheet(f"color: {COLOR_ANOMALY}; font-weight: bold;")
                    self.add_log(f"⚠️ {result_text}", error=True)
                    
                    # Start tracking anomaly duration if not already tracking
                    if self.anomaly_start_time is None:
                        self.anomaly_start_time = current_time
                        self.add_log("🕒 Starting anomaly duration tracking...", error=True)
                    
                    # Check if anomaly has persisted long enough
                    if (self.anomaly_start_time is not None and 
                        current_time - self.anomaly_start_time >= ANOMALY_THRESHOLD_TIME and
                        current_time - self.last_anomaly_notification_time >= self.notification_cooldown):
                        
                        self.add_log("📤 Preparing to send Telegram notification...", error=True)
                        message = (f"⚠️ <b>ANOMALY ALERT</b> ⚠️\n\n"
                                 f"🔍 Anomaly detected with <b>{probability:.2%}</b> confidence\n"
                                 f"⏱ Duration: <b>{current_time - self.anomaly_start_time:.1f}</b> seconds\n"
                                 f"🕒 Date: <b>{time.strftime('%d %B %Y')}</b>\n"
                                 f"⏰ Time: <b>{time.strftime('%H:%M:%S')}</b>")
                        
                        if self.send_telegram_notification(message):
                            self.last_anomaly_notification_time = current_time
                            self.add_log("✅ Telegram notification sent successfully", error=True)
                        else:
                            self.add_log("❌ Failed to send Telegram notification", error=True)
                        
            else:
                # Model yok ama veri geliyor, bunu göster
                rms = np.sqrt(np.mean(signal**2))  # RMS değeri hesapla
                result_text = f"SIGNAL RECEIVED (RMS: {rms:.4f})"
                self.result_label.setText(result_text)
                self.result_label.setStyleSheet(f"color: {COLOR_SIGNAL}; font-weight: bold;")
                self.add_log(f"No Model: {result_text}")
                
        except Exception as e:
            self.show_error(f"Error in processing completion: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def clear_plots(self):
        """Tüm grafikleri temizle"""
        try:
            # Time domain plot
            self.time_canvas.axes.clear()
            self.time_canvas.axes.set_title('Time Domain Signal')
            self.time_canvas.axes.set_xlabel('Time (s)')
            self.time_canvas.axes.set_ylabel('Amplitude')
            self.time_canvas.draw()
            
            # Frequency domain plot
            self.freq_canvas.axes.clear()
            self.freq_canvas.axes.set_title('Frequency Spectrum')
            self.freq_canvas.axes.set_xlabel('Frequency (Hz)')
            self.freq_canvas.axes.set_ylabel('Power')
            self.freq_canvas.draw()
            
            # Spectrogram - figürü tamamen temizle
            self.spec_canvas.fig.clf()
            self.spec_canvas.axes = self.spec_canvas.fig.add_subplot(111)
            
            # Tema ayarlarını tekrar uygula
            self.spec_canvas.axes.set_facecolor(COLOR_BG_DARK)
            self.spec_canvas.axes.tick_params(colors=COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.spines['bottom'].set_color(COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.spines['top'].set_color(COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.spines['left'].set_color(COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.spines['right'].set_color(COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.xaxis.label.set_color(COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.yaxis.label.set_color(COLOR_TEXT_LIGHT)
            self.spec_canvas.axes.title.set_color(COLOR_TEXT_LIGHT)
            
            self.spec_canvas.axes.set_title('Spectrogram')
            self.spec_canvas.axes.set_xlabel('Time (s)')
            self.spec_canvas.axes.set_ylabel('Frequency (Hz)')
            self.spec_canvas.fig.tight_layout()
            self.spec_canvas.draw()
        except Exception as e:
            print(f"Error clearing plots: {e}")
            import traceback
            print(traceback.format_exc())
            
    def update_plots_from_thread(self, time_data, freq_data, spec_data):
        """Thread'den gelen verilerle grafikleri güncelle"""
        try:
            # Time domain plot
            time_axis, signal = time_data
            self.time_canvas.axes.clear()
            self.time_canvas.axes.plot(time_axis, signal, color=COLOR_SIGNAL)
            self.time_canvas.axes.set_title('Time Domain Signal', color=COLOR_TEXT_LIGHT)
            self.time_canvas.axes.set_xlabel('Time (s)', color=COLOR_TEXT_LIGHT)
            self.time_canvas.axes.set_ylabel('Amplitude', color=COLOR_TEXT_LIGHT)
            self.time_canvas.axes.grid(True, color=COLOR_TEXT_LIGHT, alpha=0.3)
            
            # Tüm tick'leri beyaz yap
            self.time_canvas.axes.tick_params(axis='both', colors=COLOR_TEXT_LIGHT)
            
            # Y ekseni sınırlarını ayarla
            max_amp = max(0.1, np.max(np.abs(signal)) * 1.2)
            self.time_canvas.axes.set_ylim(-max_amp, max_amp)
            
            # Kenar çizgilerini beyaz yap
            for spine in self.time_canvas.axes.spines.values():
                spine.set_color(COLOR_TEXT_LIGHT)
                
            self.time_canvas.draw()
            
            # Frequency domain plot
            freqs, power = freq_data
            self.freq_canvas.axes.clear()
            self.freq_canvas.axes.plot(freqs, power, color=COLOR_SPECTRUM)
            self.freq_canvas.axes.set_title('Frequency Spectrum', color=COLOR_TEXT_LIGHT)
            self.freq_canvas.axes.set_xlabel('Frequency (Hz)', color=COLOR_TEXT_LIGHT)
            self.freq_canvas.axes.set_ylabel('Power', color=COLOR_TEXT_LIGHT)
            self.freq_canvas.axes.set_xlim(0, FS/2)
            
            # Tüm tick'leri beyaz yap
            self.freq_canvas.axes.tick_params(axis='both', colors=COLOR_TEXT_LIGHT)
            
            # Y ekseni sınırlarını ayarla
            max_power = max(0.1, np.max(power) * 1.2)
            self.freq_canvas.axes.set_ylim(0, max_power)
            self.freq_canvas.axes.grid(True, color=COLOR_TEXT_LIGHT, alpha=0.3)
            
            # Kenar çizgilerini beyaz yap
            for spine in self.freq_canvas.axes.spines.values():
                spine.set_color(COLOR_TEXT_LIGHT)
            
            # Highlight frequency bands
            for i, (fmin, fmax) in enumerate(BANDS):
                band_color = plt.cm.viridis(i/len(BANDS))
                self.freq_canvas.axes.axvspan(fmin, fmax, alpha=0.2, color=band_color)
            self.freq_canvas.draw()
            
            # Spectrogram
            if self.show_spectrogram.isChecked():
                f, t, Sxx = spec_data
                
                # Figürü tamamen temizle (colorbar dahil)
                self.spec_canvas.fig.clf()
                
                # Yeni axes oluştur
                self.spec_canvas.axes = self.spec_canvas.fig.add_subplot(111)
                
                # Tema ayarlarını tekrar uygula
                self.spec_canvas.axes.set_facecolor(COLOR_BG_DARK)
                self.spec_canvas.axes.tick_params(axis='both', colors=COLOR_TEXT_LIGHT)
                
                # Kenar çizgilerini beyaz yap
                for spine in self.spec_canvas.axes.spines.values():
                    spine.set_color(COLOR_TEXT_LIGHT)
                
                # Spektrogramı çiz
                im = self.spec_canvas.axes.pcolormesh(t, f, Sxx, shading='gouraud', cmap='inferno')
                self.spec_canvas.axes.set_title('Spectrogram', color=COLOR_TEXT_LIGHT)
                self.spec_canvas.axes.set_xlabel('Time (s)', color=COLOR_TEXT_LIGHT)
                self.spec_canvas.axes.set_ylabel('Frequency (Hz)', color=COLOR_TEXT_LIGHT)
                self.spec_canvas.axes.set_ylim(0, FS/2)
                
                # Colorbar ekle ve renklerini ayarla
                cbar = self.spec_canvas.fig.colorbar(im, ax=self.spec_canvas.axes)
                cbar.set_label('Power (dB)', color=COLOR_TEXT_LIGHT)
                cbar.ax.yaxis.set_tick_params(colors=COLOR_TEXT_LIGHT)
                cbar.ax.yaxis.label.set_color(COLOR_TEXT_LIGHT)
                
                # Colorbar'ın tick label'larını beyaz yap
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLOR_TEXT_LIGHT)
                
                # Tight layout uygula ve çiz
                self.spec_canvas.fig.tight_layout()
                self.spec_canvas.draw()
                
            # Durum çubuğunu güncelle
            self.statusBar().showMessage(f"Plots updated with {len(signal)} samples", 1000)
            
        except Exception as e:
            print(f"Error updating plots from thread: {e}")
            import traceback
            print(traceback.format_exc())
            
    def plot_feature_importance(self, X_test, y_test):
        """Plot feature importance based on feature effect on predictions"""
        try:
            self.features_canvas.axes.clear()
            self.features_canvas.axes.set_facecolor(COLOR_BG_DARK)
            self.features_canvas.fig.set_facecolor(COLOR_BG_DARK)
            
            if self.model is None:
                return
                
            # For SVM, we'll use a simple approach: measure change in prediction when each feature changes
            importance = np.zeros(X_test.shape[1])
            
            # Calculate average impact of each feature
            n_samples = min(100, X_test.shape[0])  # Limit to 100 samples for speed
            for i in range(min(X_test.shape[1], len(self.feature_names))):
                X_modified = X_test[:n_samples].copy()
                # Increase feature by 1 standard deviation
                X_modified[:, i] += 1.0
                # Check how predictions change
                original_preds = self.model.predict_proba(X_test[:n_samples])[:, 1]
                modified_preds = self.model.predict_proba(X_modified)[:, 1]
                # Average absolute change in prediction
                importance[i] = np.mean(np.abs(modified_preds - original_preds))
            
            # Plot feature importance
            feature_labels = self.feature_names[:len(importance)]
            colors = [COLOR_NORMAL if val < np.mean(importance) else COLOR_ANOMALY for val in importance]
            
            bars = self.features_canvas.axes.bar(feature_labels, importance, color=colors)
            self.features_canvas.axes.set_title('Feature Importance', color=COLOR_TEXT_LIGHT)
            self.features_canvas.axes.set_ylabel('Importance Score', color=COLOR_TEXT_LIGHT)
            self.features_canvas.axes.tick_params(axis='both', colors=COLOR_TEXT_LIGHT)
            
            # X axis labels
            self.features_canvas.axes.tick_params(axis='x', rotation=45)
            plt.setp(self.features_canvas.axes.get_xticklabels(), color=COLOR_TEXT_LIGHT)
            
            # Add a horizontal line at the mean importance
            self.features_canvas.axes.axhline(y=np.mean(importance), 
                                           color=COLOR_TEXT_LIGHT, 
                                           linestyle='--', 
                                           alpha=0.7)
            
            # Set spines color
            for spine in self.features_canvas.axes.spines.values():
                spine.set_color(COLOR_TEXT_LIGHT)
            
            self.features_canvas.fig.tight_layout()
            self.features_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error plotting feature importance: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def train_model(self):
        """Train the anomaly detection model"""
        try:
            normal_dir = self.normal_path.text()
            anomaly_dir = self.anomaly_path.text()
            
            if not normal_dir or not anomaly_dir:
                self.show_error("Please select both normal and anomaly data directories")
                return
                
            if not os.path.exists(normal_dir) or not os.path.exists(anomaly_dir):
                self.show_error("One or both directories do not exist")
                return
                
            # Clear previous results
            self.result_text.clear()
            
            # Load dataset
            self.result_text.append("Loading dataset...")
            QApplication.processEvents()
            
            X, y = load_dataset(normal_dir, anomaly_dir)
            
            if len(X) == 0 or len(y) == 0:
                self.show_error("No valid data found in the selected directories")
                return
                
            self.result_text.append(f"Loaded {len(X)} samples")
            QApplication.processEvents()
            
            # Update feature names
            self.feature_names = ['RMS', 'ZCR', 'Peak', 'Crest', 'Centroid', 'Bandwidth', 'Rolloff', 'Flatness'] + [f'Band {i+1}' for i in range(len(BANDS))]
            
            # Split into train/test sets
            test_size = self.test_size_spin.value()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.result_text.append("Training model...")
            QApplication.processEvents()
            
            C = self.c_param.value()
            self.model = SVC(kernel='rbf', C=C, gamma='auto', probability=True, class_weight='balanced')
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)
            
            # Display results
            report = classification_report(y_test, y_pred, digits=4)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            self.result_text.append("\n====== Classification Report ======")
            self.result_text.append(report)
            self.result_text.append("\n====== Confusion Matrix ======")
            self.result_text.append(str(conf_matrix))
            
            # Add validation information
            self.result_text.append("\n====== Model Validation ======")
            self.result_text.append(f"Normal samples correctly classified: {100 * (y_test[y_test==0] == y_pred[y_test==0]).mean():.2f}%")
            self.result_text.append(f"Anomaly samples correctly classified: {100 * (y_test[y_test==1] == y_pred[y_test==1]).mean():.2f}%")
            
            # Plot feature importance
            self.plot_feature_importance(X_test_scaled, y_test)
            
            self.statusBar().showMessage("Model trained successfully", 3000)
            
        except Exception as e:
            self.show_error(f"Error during training: {str(e)}")
            import traceback
            self.result_text.append("\n====== Error Traceback ======")
            self.result_text.append(traceback.format_exc())

    def process_raw_data(self, raw_data):
        """Process raw data received from serial port"""
        try:
            # Tampon boyutunu kontrol et
            if len(self.raw_data_buffer) >= 100:
                self.raw_data_buffer.popleft()  # En eski veriyi çıkar
            
            # Yeni veriyi ekle
            self.raw_data_buffer.append(raw_data)
            
            # TextEdit'i güncelle (son 10 veriyi göster)
            self.raw_data_text.clear()
            for i, data in enumerate(list(self.raw_data_buffer)[-10:]):
                self.raw_data_text.append(f"{i+1}: {data}")
            
            # Scroll to bottom
            self.raw_data_text.verticalScrollBar().setValue(
                self.raw_data_text.verticalScrollBar().maximum()
            )
            
            # Son ADC değerini güncelle (eğer varsa)
            if self.serial_worker.raw_data_buffer:
                last_value = self.serial_worker.raw_data_buffer[-1]
                self.add_log(f"Received ADC value: {last_value}")
                
                # Veri alındığını göstermek için durum çubuğunu güncelle
                self.statusBar().showMessage(f"Receiving data: Last ADC value = {last_value}", 1000)
                
                # Veri geldiğini göstermek için etiket rengini değiştir
                if self.result_label.text() == "No Data" or self.result_label.text() == "Waiting for data...":
                    self.result_label.setText("DATA RECEIVING")
                    self.result_label.setStyleSheet(f"color: {COLOR_SIGNAL}; font-weight: bold;")
        except Exception as e:
            self.show_error(f"Error processing raw data: {str(e)}")

    def load_sample_data(self, is_anomaly):
        """Load and display sample fan data"""
        try:
            # Stop any existing playback
            self.sample_timer.stop()
            self.current_sample_index = 0
            
            # Load sample data
            data_type = "anomaly" if is_anomaly else "normal"
            sample_file = f"data/{data_type}/{data_type}_fan_1.csv"
            
            if not os.path.exists(sample_file):
                self.show_error(f"Sample data not found. Please run generate_fan_data.py first.")
                return
                
            df = pd.read_csv(sample_file)
            self.sample_data = df['amplitude'].values
            
            # Start playback timer (update every 100ms)
            self.sample_timer.start(100)
            
            # Update status
            status = "Anomaly" if is_anomaly else "Normal"
            self.statusBar().showMessage(f"Playing {status} fan sample data")
            
        except Exception as e:
            self.show_error(f"Error loading sample data: {str(e)}")
            
    def update_sample_display(self):
        """Update display with next chunk of sample data"""
        try:
            if not len(self.sample_data):
                self.sample_timer.stop()
                return
                
            # Get next chunk of data (1000 samples)
            chunk_size = 1000
            start_idx = self.current_sample_index
            end_idx = start_idx + chunk_size
            
            if start_idx >= len(self.sample_data):
                # Reset to beginning when reaching the end
                self.current_sample_index = 0
                start_idx = 0
                end_idx = chunk_size
                
            chunk = self.sample_data[start_idx:end_idx]
            self.current_sample_index = end_idx
            
            # Process the chunk as if it was received from serial
            self.process_new_data(chunk)
            
        except Exception as e:
            self.show_error(f"Error updating sample display: {str(e)}")
            self.sample_timer.stop()


# Add this at the end of your fan_anomaly_detector.py file

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()