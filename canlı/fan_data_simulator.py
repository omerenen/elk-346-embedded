#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fan_data_simulator.py

Simulates fan data for testing the anomaly detection system:
1) Simulates normal fan operation with predominant frequencies
2) Simulates anomalous fan operation with added frequencies and noise
3) Outputs CSV files for both normal and anomalous data
4) Can also simulate real-time data via serial port
"""

import os
import numpy as np
import pandas as pd
import time
import argparse
import serial
from tqdm import tqdm

# Parameters
SAMPLE_RATE = 8000  # Hz
DURATION = 1.0      # seconds
ADC_BITS = 12
VREF = 3.3

# Fan simulation parameters
NORMAL_FREQUENCIES = [50, 100, 200]  # Base frequencies for normal fan (Hz)
ANOMALY_FREQUENCIES = [300, 500, 770]  # Additional frequencies for anomalous fan (Hz)


def generate_normal_sample(duration=DURATION, fs=SAMPLE_RATE, noise_level=0.05):
    """Generate a normal fan signal"""
    num_samples = int(duration * fs)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Create base signal with normal frequencies
    signal = np.zeros(num_samples)
    for freq in NORMAL_FREQUENCIES:
        # Add base frequency with random amplitude between 0.1 and 0.3
        amplitude = 0.1 + 0.2 * np.random.random()
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    signal += noise_level * np.random.randn(num_samples)
    
    # Normalize to [-1, 1] range
    signal = signal / np.max(np.abs(signal))
    
    return signal


def generate_anomaly_sample(duration=DURATION, fs=SAMPLE_RATE, noise_level=0.15, 
                           anomaly_strength=0.5):
    """Generate an anomalous fan signal"""
    num_samples = int(duration * fs)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Start with a normal signal
    signal = generate_normal_sample(duration, fs, noise_level)
    
    # Add anomaly frequencies
    for freq in ANOMALY_FREQUENCIES:
        # Random amplitude for the anomaly frequency
        amplitude = anomaly_strength * (0.1 + 0.3 * np.random.random())
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some extra noise bursts
    burst_location = np.random.randint(0, num_samples - int(0.1*fs))
    burst_duration = int(0.05 * fs)  # 50ms burst
    signal[burst_location:burst_location+burst_duration] += 0.2 * np.random.randn(burst_duration)
    
    # Normalize to [-1, 1] range
    signal = signal / np.max(np.abs(signal))
    
    return signal


def signal_to_adc(signal, bits=ADC_BITS, vref=VREF):
    """Convert normalized signal to ADC counts"""
    # Scale from [-1, 1] to [0, Vref]
    voltage = (signal + 1) * vref / 2
    # Convert to ADC counts
    counts = np.round(voltage / vref * (2**bits - 1))
    # Clip to valid range
    return np.clip(counts, 0, 2**bits - 1).astype(np.uint16)


def generate_dataset(num_normal=100, num_anomaly=30, output_dir="./data"):
    """Generate a dataset of normal and anomalous fan signals"""
    # Create output directories
    normal_dir = os.path.join(output_dir, 'normal')
    anomaly_dir = os.path.join(output_dir, 'anomaly')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)
    
    # Generate normal samples
    print(f"Generating {num_normal} normal samples...")
    for i in tqdm(range(num_normal)):
        signal = generate_normal_sample()
        adc_values = signal_to_adc(signal)
        
        # Create dataframe with time and ADC values
        t = np.linspace(0, DURATION, len(signal), endpoint=False)
        df = pd.DataFrame({
            'time': t,
            'adc_value': adc_values
        })
        
        # Save to CSV
        filename = os.path.join(normal_dir, f'normal_sample_{i+1:03d}.csv')
        df.to_csv(filename, index=False)
    
    # Generate anomaly samples
    print(f"Generating {num_anomaly} anomaly samples...")
    for i in tqdm(range(num_anomaly)):
        signal = generate_anomaly_sample()
        adc_values = signal_to_adc(signal)
        
        # Create dataframe with time and ADC values
        t = np.linspace(0, DURATION, len(signal), endpoint=False)
        df = pd.DataFrame({
            'time': t,
            'adc_value': adc_values
        })
        
        # Save to CSV
        filename = os.path.join(anomaly_dir, f'anomaly_sample_{i+1:03d}.csv')
        df.to_csv(filename, index=False)
    
    print(f"Dataset generated in {output_dir}")


def simulate_serial(port, baudrate=115200, duration_seconds=60, probability_anomaly=0.2):
    """Simulate serial data stream for real-time processing"""
    try:
        ser = serial.Serial(port, baudrate)
        print(f"Serial port {port} opened at {baudrate} baud")
        
        # Calculate samples per window
        samples_per_window = int(DURATION * SAMPLE_RATE)
        
        # Run for specified duration
        start_time = time.time()
        sample_count = 0
        window_count = 0
        
        print(f"Sending simulated fan data for {duration_seconds} seconds...")
        
        while (time.time() - start_time) < duration_seconds:
            # Decide if this window will be normal or anomalous
            is_anomaly = np.random.random() < probability_anomaly
            
            # Generate appropriate signal
            if is_anomaly:
                signal = generate_anomaly_sample()
                print(f"Window {window_count+1}: Sending ANOMALY data")
            else:
                signal = generate_normal_sample()
                print(f"Window {window_count+1}: Sending NORMAL data")
                
            # Convert to ADC values
            adc_values = signal_to_adc(signal)
            
            # Send values one by one with a small delay
            for value in adc_values:
                ser.write(f"{value}\n".encode())
                sample_count += 1
                
                # Add a small delay to simulate sampling rate
                time.sleep(1.0 / SAMPLE_RATE)
                
            window_count += 1
            
        print(f"Simulation complete. Sent {sample_count} samples in {window_count} windows.")
        ser.close()
        
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
        if 'ser' in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fan data simulator for anomaly detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate dataset command
    gen_parser = subparsers.add_parser("generate", help="Generate a dataset of CSV files")
    gen_parser.add_argument("--normal", type=int, default=100, help="Number of normal samples")
    gen_parser.add_argument("--anomaly", type=int, default=30, help="Number of anomalous samples")
    gen_parser.add_argument("--output", type=str, default="./data", help="Output directory")
    
    # Simulate serial command
    sim_parser = subparsers.add_parser("serial", help="Simulate serial data stream")
    sim_parser.add_argument("port", type=str, help="Serial port (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux)")
    sim_parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    sim_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    sim_parser.add_argument("--anomaly-prob", type=float, default=0.2, help="Probability of anomaly")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_dataset(args.normal, args.anomaly, args.output)
    elif args.command == "serial":
        simulate_serial(args.port, args.baudrate, args.duration, args.anomaly_prob)
    else:
        parser.print_help()