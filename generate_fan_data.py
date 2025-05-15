import numpy as np
import pandas as pd
import os
from scipy import signal

def generate_fan_signal(duration, fs, base_freq, noise_level=0.1, anomaly=False):
    """
    Generate synthetic fan sound data
    
    Parameters:
    - duration: Signal duration in seconds
    - fs: Sampling frequency in Hz
    - base_freq: Base frequency of the fan (e.g., 50Hz for a typical fan)
    - noise_level: Background noise level (0-1)
    - anomaly: If True, generate anomaly signal
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Base fan signal (fundamental frequency)
    signal_clean = np.sin(2 * np.pi * base_freq * t)
    
    # Add harmonics
    for i in range(2, 5):
        harmonic_amp = 1.0 / i
        signal_clean += harmonic_amp * np.sin(2 * np.pi * base_freq * i * t)
    
    # Add some mechanical variation (slight frequency modulation)
    fm = 2  # 2 Hz modulation
    signal_clean *= (1 + 0.1 * np.sin(2 * np.pi * fm * t))
    
    # Normalize
    signal_clean = signal_clean / np.max(np.abs(signal_clean))
    
    # Add background noise
    noise = np.random.normal(0, noise_level, len(t))
    signal_final = signal_clean + noise
    
    if anomaly:
        # Add bearing fault simulation (periodic impulses)
        fault_freq = 20  # Fault frequency in Hz
        fault_times = np.arange(0, duration, 1/fault_freq)
        for fault_time in fault_times:
            idx = int(fault_time * fs)
            if idx < len(signal_final):
                # Add an impulse and its decay
                decay_length = int(0.02 * fs)  # 20ms decay
                decay = np.exp(-np.arange(decay_length) / (0.005 * fs))
                end_idx = min(idx + decay_length, len(signal_final))
                signal_final[idx:end_idx] += 0.5 * decay[:end_idx-idx]
        
        # Add random amplitude modulation (unbalance simulation)
        unbalance = 0.3 * np.sin(2 * np.pi * 3 * t)  # 3 Hz unbalance
        signal_final *= (1 + unbalance)
        
        # Add random high-frequency components (bearing wear simulation)
        bearing_wear = 0.2 * np.sin(2 * np.pi * 500 * t)
        signal_final += bearing_wear
    
    # Final normalization
    signal_final = signal_final / np.max(np.abs(signal_final))
    
    return signal_final

def save_data(signal, filename, fs):
    """Save signal data to CSV file"""
    t = np.arange(len(signal)) / fs
    df = pd.DataFrame({
        'time': t,
        'amplitude': signal
    })
    df.to_csv(filename, index=False)

def main():
    # Parameters
    fs = 8000  # Sampling frequency (Hz)
    duration = 1.0  # Duration of each sample (seconds)
    base_freq = 50  # Base frequency of fan (Hz)
    n_samples = 10  # Number of samples for each condition
    
    # Create directories
    os.makedirs('data/normal', exist_ok=True)
    os.makedirs('data/anomaly', exist_ok=True)
    
    # Generate normal fan data
    for i in range(n_samples):
        signal = generate_fan_signal(duration, fs, base_freq, noise_level=0.1, anomaly=False)
        save_data(signal, f'data/normal/normal_fan_{i+1}.csv', fs)
    
    # Generate anomaly fan data
    for i in range(n_samples):
        signal = generate_fan_signal(duration, fs, base_freq, noise_level=0.15, anomaly=True)
        save_data(signal, f'data/anomaly/anomaly_fan_{i+1}.csv', fs)
    
    print(f"Generated {n_samples} normal and {n_samples} anomaly samples")
    print("Data saved in 'data/normal' and 'data/anomaly' directories")

if __name__ == "__main__":
    main() 