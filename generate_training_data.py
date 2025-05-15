#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy import signal as sg  # signal modülünü sg olarak yeniden adlandır

# Örnekleme parametreleri
FS = 8000  # Örnekleme frekansı (Hz)
DURATION = 1.0  # Her örnek için süre (saniye)
N_SAMPLES = int(FS * DURATION)  # Örnek başına nokta sayısı

# Fan karakteristikleri
NORMAL_FREQ = 50  # Normal fan frekansı (Hz)
BEARING_FREQ = [120, 180]  # Rulman arıza frekansları (Hz)
IMBALANCE_FREQ = [30, 70]  # Dengesizlik frekansları (Hz)

def generate_normal_fan(n_samples=N_SAMPLES, noise_level=0.1):
    """Normal çalışan fan sinyali üret"""
    t = np.linspace(0, DURATION, n_samples)
    
    # Ana fan frekansı
    main_signal = np.sin(2 * np.pi * NORMAL_FREQ * t)
    
    # Harmonikler ekle (2x ve 3x frekanslar)
    harmonics = (0.5 * np.sin(4 * np.pi * NORMAL_FREQ * t) + 
                0.25 * np.sin(6 * np.pi * NORMAL_FREQ * t))
    
    # Rastgele gürültü ekle
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Tüm bileşenleri birleştir
    signal = main_signal + harmonics + noise
    
    # Normalize et (-1 ile 1 arasına)
    signal = signal / np.max(np.abs(signal))
    
    return signal

def generate_anomaly_fan(n_samples=N_SAMPLES, noise_level=0.2, anomaly_type='random'):
    """Anormal çalışan fan sinyali üret"""
    t = np.linspace(0, DURATION, n_samples)
    
    # Temel sinyal (zayıflamış ana frekans)
    main_signal = 0.7 * np.sin(2 * np.pi * NORMAL_FREQ * t)
    
    if anomaly_type == 'bearing':
        # Rulman arızası: Yüksek frekanslı titreşimler
        for freq in BEARING_FREQ:
            main_signal += 0.4 * np.sin(2 * np.pi * freq * t)
        
        # Genlik modülasyonu ekle
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 10 * t)  # 10 Hz modülasyon
        main_signal *= modulation
        
    elif anomaly_type == 'imbalance':
        # Dengesizlik: Düşük frekanslı salınımlar
        for freq in IMBALANCE_FREQ:
            main_signal += 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Rastgele genlik değişimleri
        amplitude_var = 1 + 0.2 * np.random.randn(n_samples)
        main_signal *= amplitude_var
        
    else:  # random anomalies
        # Rastgele frekans bileşenleri
        random_freqs = np.random.uniform(20, 200, 5)  # 5 rastgele frekans
        for freq in random_freqs:
            main_signal += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Daha yüksek gürültü seviyesi
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Tüm bileşenleri birleştir
    signal = main_signal + noise
    
    # Normalize et (-1 ile 1 arasına)
    signal = signal / np.max(np.abs(signal))
    
    return signal

def add_transients(input_signal, n_transients=3):
    """Sinyale geçici bozulmalar ekle"""
    signal = input_signal.copy()
    n_samples = len(signal)
    
    for _ in range(n_transients):
        # Rastgele bir pozisyon seç
        pos = np.random.randint(0, n_samples)
        # Geçici bozulma genliği
        amplitude = np.random.uniform(0.5, 1.0)
        # Geçici bozulma uzunluğu
        length = np.random.randint(10, 50)
        
        # Gaussian pencere oluştur
        gaussian_window = sg.windows.gaussian(length, std=length/6.0)
        
        # Geçici bozulmayı ekle
        start = max(0, pos - length//2)
        end = min(n_samples, pos + length//2)
        window_length = end - start
        if window_length > 0:
            signal[start:end] += amplitude * gaussian_window[:window_length]
    
    return signal

def generate_dataset(n_normal=1000, n_anomaly=1000):
    """Eğitim ve test için veri seti oluştur"""
    # Klasörleri oluştur
    os.makedirs("data/normal", exist_ok=True)
    os.makedirs("data/anomaly", exist_ok=True)
    
    # Normal örnekler
    print(f"Generating {n_normal} normal samples...")
    for i in range(n_normal):
        signal = generate_normal_fan()
        
        # Rastgele varyasyonlar ekle
        noise_level = np.random.uniform(0.05, 0.15)
        signal += np.random.normal(0, noise_level, len(signal))
        
        # Normalize et
        signal = signal / np.max(np.abs(signal))
        
        # Kaydet
        df = pd.DataFrame({
            'time': np.linspace(0, DURATION, N_SAMPLES),
            'amplitude': signal
        })
        df.to_csv(f"data/normal/normal_fan_{i+1}.csv", index=False)
        
        if (i+1) % 100 == 0:
            print(f"Generated {i+1} normal samples")
    
    # Anormal örnekler
    print(f"\nGenerating {n_anomaly} anomaly samples...")
    anomaly_types = ['bearing', 'imbalance', 'random']
    for i in range(n_anomaly):
        # Rastgele bir anomali tipi seç
        anomaly_type = np.random.choice(anomaly_types)
        signal = generate_anomaly_fan(anomaly_type=anomaly_type)
        
        # Rastgele geçici bozulmalar ekle
        if np.random.random() < 0.3:  # %30 olasılıkla
            signal = add_transients(signal)
        
        # Normalize et
        signal = signal / np.max(np.abs(signal))
        
        # Kaydet
        df = pd.DataFrame({
            'time': np.linspace(0, DURATION, N_SAMPLES),
            'amplitude': signal
        })
        df.to_csv(f"data/anomaly/anomaly_fan_{i+1}.csv", index=False)
        
        if (i+1) % 100 == 0:
            print(f"Generated {i+1} anomaly samples")

if __name__ == "__main__":
    print("Generating synthetic fan data for training...")
    generate_dataset(n_normal=1000, n_anomaly=1000)
    print("\nData generation complete!")
    print("Normal samples saved in: data/normal/")
    print("Anomaly samples saved in: data/anomaly/") 