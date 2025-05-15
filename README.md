# Fan Anomaly Detection System 🌪️

Real-time fan anomaly detection system using machine learning and signal processing techniques. The system monitors fan behavior through ADC readings and alerts users via Telegram when anomalies are detected.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## 🌟 Features

- **Real-time Monitoring** 📊
  - ADC data acquisition through serial port
  - Time-domain signal visualization
  - Frequency spectrum analysis
  - Spectrogram display
  - Raw data logging

- **Advanced Signal Processing** 🔍
  - Feature extraction from time and frequency domains
  - Band energy analysis
  - Spectral characteristics computation
  - Real-time signal processing

- **Machine Learning** 🤖
  - SVM-based anomaly detection
  - Model training interface
  - Feature importance visualization
  - Performance metrics display

- **Alert System** ⚠️
  - Real-time anomaly detection
  - Telegram notifications
  - Configurable alert thresholds
  - Detailed anomaly reporting

- **User Interface** 💻
  - Dark theme modern interface
  - Multiple visualization tabs
  - Interactive controls
  - Real-time status updates

## 📋 Requirements

```bash
# Core dependencies
numpy
pandas
scikit-learn
scipy
matplotlib
PyQt5

# Communication
pyserial
requests

# Optional (for synthetic data generation)
python-telegram-bot
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/omerenen/fan-anomaly-detector.git
cd fan-anomaly-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Telegram bot (optional):
   - Create a new bot using [@BotFather](https://t.me/botfather)
   - Update `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the code

## 💡 Usage

1. **Generate Training Data**:
```bash
python generate_training_data.py
```

2. **Run the Application**:
```bash
python fan_anomaly_detector.py
```

3. **Training the Model**:
   - Navigate to "Training & Evaluation" tab
   - Select normal and anomaly data folders
   - Adjust training parameters
   - Click "Train Model"

4. **Real-time Monitoring**:
   - Connect to serial port or use synthetic data
   - Monitor real-time visualizations
   - Receive Telegram alerts for anomalies

## 📊 System Architecture

```
├── Data Acquisition
│   ├── Serial Port Reading
│   └── Synthetic Data Generation
│
├── Signal Processing
│   ├── Feature Extraction
│   ├── Spectral Analysis
│   └── Band Energy Computation
│
├── Machine Learning
│   ├── SVM Model
│   ├── Feature Scaling
│   └── Anomaly Detection
│
└── User Interface
    ├── Real-time Plots
    ├── Control Panel
    └── Alert System
```

## ⚙️ Configuration

Key parameters can be adjusted in the settings:

- Sampling frequency (default: 8000 Hz)
- ADC resolution (default: 12 bits)
- Window size (default: 1 second)
- Alert threshold (default: 1 second)
- Notification cooldown (default: 60 seconds)

## 🔧 Customization

The system can be customized for different scenarios:

- Adjust frequency bands for analysis
- Modify feature extraction parameters
- Change machine learning model parameters
- Customize alert thresholds and conditions

