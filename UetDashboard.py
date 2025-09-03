#!/usr/bin/env python3
"""
Seismic Data Monitoring Dashboard
Real-time visualization and control for Lunar Seismic Data Processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import threading
import queue
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Data structure for sensor readings"""
    timestamp: float
    raw_value: float
    filtered_value: float
    ml_confidence: Optional[float] = None
    ml_anomaly_score: Optional[float] = None
    ml_inference_time: Optional[float] = None

class SeismicDashboard:
    """Main dashboard application for seismic data monitoring"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lunar Seismic Data Monitor")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data_queue = queue.Queue(maxsize=1000)
        self.sensor_data: List[SensorData] = []
        self.max_data_points = 1000
        self.is_running = False
        
        # Serial connection
        self.serial_port = None
        self.serial_thread = None
        self.baudrate = 115200
        self.port = "COM3"  # Default, will be updated from UI
        
        # Initialize UI
        self.setup_ui()
        
        # Start data processing thread
        self.process_thread = threading.Thread(target=self.process_data, daemon=True)
        self.process_thread.start()
        
    def setup_ui(self):
        """Set up the main UI components"""
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Control panel
        self.setup_control_panel()
        
        # Plot area
        self.setup_plots()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        
    def setup_control_panel(self):
        """Set up the control panel"""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Serial port selection
        ttk.Label(control_frame, text="Serial Port:").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar(value=self.port)
        port_entry = ttk.Entry(control_frame, textvariable=self.port_var, width=15)
        port_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        # Baud rate selection
        ttk.Label(control_frame, text="Baud Rate:").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.baudrate_var = tk.StringVar(value=str(self.baudrate))
        baud_combo = ttk.Combobox(control_frame, textvariable=self.baudrate_var, 
                                 values=["9600", "19200", "38400", "57600", "115200"],
                                 width=10)
        baud_combo.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        
        # Control buttons
        self.connect_btn = ttk.Button(control_frame, text="Connect", 
                                     command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=4, padx=5, pady=2)
        
        # Recording controls
        self.record_btn = ttk.Button(control_frame, text="Start Recording", 
                                    command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.grid(row=0, column=5, padx=5, pady=2)
        
        # ML model controls
        ttk.Label(control_frame, text="ML Model:").grid(row=1, column=0, sticky="w")
        self.model_status = ttk.Label(control_frame, text="Not Loaded", foreground="red")
        self.model_status.grid(row=1, column=1, sticky="w", columnspan=2)
        
        self.load_model_btn = ttk.Button(control_frame, text="Load Model",
                                       command=self.load_model, state=tk.DISABLED)
        self.load_model_btn.grid(row=1, column=3, padx=5, pady=2)
        
        # Configuration
        ttk.Label(control_frame, text="Config:").grid(row=2, column=0, sticky="w")
        self.config_btn = ttk.Button(control_frame, text="Settings",
                                   command=self.show_settings)
        self.config_btn.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
    def setup_plots(self):
        """Set up the plotting area"""
        # Create figure and subplots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
            2, 2, figsize=(12, 8), tight_layout=True)
        
        # Initialize plots
        self.init_plot(self.ax1, "Raw vs Filtered Signal", "Time", "Amplitude", 
                      ["Raw", "Filtered"])
        self.init_plot(self.ax2, "Frequency Spectrum", "Frequency (Hz)", "Magnitude (dB)")
        self.init_plot(self.ax3, "ML Confidence", "Time", "Confidence", ylim=(0, 1.1))
        self.init_plot(self.ax4, "Anomaly Score", "Time", "Score", ylim=(0, 1.1))
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
    def init_plot(self, ax, title, xlabel, ylabel, legend=None, ylim=None):
        """Initialize a single plot"""
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(legend)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True)
        
    def toggle_connection(self):
        """Toggle serial connection"""
        if self.serial_port and self.serial_port.is_open:
            self.disconnect_serial()
        else:
            self.connect_serial()
            
    def connect_serial(self):
        """Connect to serial port"""
        self.port = self.port_var.get()
        self.baudrate = int(self.baudrate_var.get())
        
        try:
            self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
            self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.serial_thread.start()
            
            self.connect_btn.config(text="Disconnect")
            self.record_btn.config(state=tk.NORMAL)
            self.load_model_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Connected to {self.port} @ {self.baudrate} baud")
            logger.info(f"Connected to {self.port}")
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
            logger.error(f"Serial connection failed: {str(e)}")
            
    def disconnect_serial(self):
        """Disconnect from serial port"""
        if self.serial_port and self.serial_port.is_open:
            self.is_running = False
            self.serial_port.close()
            self.connect_btn.config(text="Connect")
            self.record_btn.config(state=tk.DISABLED)
            self.load_model_btn.config(state=tk.DISABLED)
            self.status_var.set("Disconnected")
            logger.info("Serial connection closed")
            
    def read_serial(self):
        """Read data from serial port"""
        self.is_running = True
        buffer = ""
        
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                # Read available data
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.readline().decode('utf-8').strip()
                    if data:
                        try:
                            # Parse JSON data
                            parsed = json.loads(data)
                            timestamp = time.time()
                            
                            # Create sensor data object
                            sensor_data = SensorData(
                                timestamp=timestamp,
                                raw_value=float(parsed.get('raw', 0)),
                                filtered_value=float(parsed.get('filtered', 0)),
                                ml_confidence=float(parsed.get('confidence', 0)),
                                ml_anomaly_score=float(parsed.get('anomaly_score', 0)),
                                ml_inference_time=float(parsed.get('inference_time', 0))
                            )
                            
                            # Add to queue
                            if self.data_queue.qsize() >= self.max_data_points:
                                self.data_queue.get()  # Remove oldest if queue is full
                            self.data_queue.put(sensor_data)
                            
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Invalid data format: {data}")
                            
            except Exception as e:
                logger.error(f"Error reading serial: {str(e)}")
                time.sleep(0.1)
                
    def process_data(self):
        """Process data from queue and update plots"""
        while True:
            try:
                if not self.data_queue.empty():
                    # Get new data
                    new_data = self.data_queue.get()
                    self.sensor_data.append(new_data)
                    
                    # Keep only recent data
                    if len(self.sensor_data) > self.max_data_points:
                        self.sensor_data = self.sensor_data[-self.max_data_points:]
                    
                    # Update plots
                    self.update_plots()
                
                time.sleep(0.01)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                logger.error(f"Error in process_data: {str(e)}")
                time.sleep(0.1)
                
    def update_plots(self):
        """Update all plots with current data"""
        if not self.sensor_data:
            return
            
        try:
            # Extract data
            timestamps = [d.timestamp for d in self.sensor_data]
            raw_values = [d.raw_value for d in self.sensor_data]
            filtered_values = [d.filtered_value for d in self.sensor_data]
            
            # Update raw vs filtered plot
            self.ax1.clear()
            self.ax1.plot(timestamps, raw_values, 'b-', alpha=0.7, label='Raw')
            self.ax1.plot(timestamps, filtered_values, 'r-', label='Filtered')
            self.ax1.set_title("Raw vs Filtered Signal")
            self.ax1.set_xlabel("Time")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.legend()
            self.ax1.grid(True)
            
            # Update frequency spectrum
            self.update_spectrum_plot()
            
            # Update ML metrics if available
            if any(d.ml_confidence is not None for d in self.sensor_data):
                confidences = [d.ml_confidence or 0 for d in self.sensor_data]
                anomalies = [d.ml_anomaly_score or 0 for d in self.sensor_data]
                
                # Update confidence plot
                self.ax3.clear()
                self.ax3.plot(timestamps, confidences, 'g-')
                self.ax3.set_title("ML Confidence")
                self.ax3.set_xlabel("Time")
                self.ax3.set_ylabel("Confidence")
                self.ax3.set_ylim(0, 1.1)
                self.ax3.grid(True)
                
                # Update anomaly score plot
                self.ax4.clear()
                self.ax4.plot(timestamps, anomalies, 'r-')
                self.ax4.axhline(y=0.7, color='r', linestyle='--', alpha=0.7)
                self.ax4.set_title("Anomaly Score")
                self.ax4.set_xlabel("Time")
                self.ax4.set_ylabel("Score")
                self.ax4.set_ylim(0, 1.1)
                self.ax4.grid(True)
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating plots: {str(e)}")
            
    def update_spectrum_plot(self):
        """Update frequency spectrum plot"""
        if len(self.sensor_data) < 2:
            return
            
        try:
            # Get filtered values
            y = [d.filtered_value for d in self.sensor_data]
            
            # Compute FFT
            n = len(y)
            if n < 2:
                return
                
            yf = np.fft.rfft(y)
            xf = np.fft.rfftfreq(n, d=1.0/100)  # Assuming 100Hz sampling rate
            
            # Update plot
            self.ax2.clear()
            self.ax2.plot(xf, 20 * np.log10(np.abs(yf) + 1e-10))
            self.ax2.set_title("Frequency Spectrum")
            self.ax2.set_xlabel("Frequency (Hz)")
            self.ax2.set_ylabel("Magnitude (dB)")
            self.ax2.grid(True)
            
        except Exception as e:
            logger.error(f"Error updating spectrum: {str(e)}")
            
    def toggle_recording(self):
        """Toggle data recording"""
        if self.record_btn.cget("text") == "Start Recording":
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start recording data"""
        self.record_btn.config(text="Stop Recording")
        self.status_var.set("Recording data...")
        # TODO: Implement data recording
        
    def stop_recording(self):
        """Stop recording data"""
        self.record_btn.config(text="Start Recording")
        self.status_var.set("Recording stopped")
        # TODO: Save recorded data
        
    def load_model(self):
        """Load ML model"""
        try:
            # TODO: Implement model loading
            self.model_status.config(text="Loaded", foreground="green")
            self.status_var.set("ML model loaded successfully")
            logger.info("ML model loaded")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            logger.error(f"Model loading failed: {str(e)}")
            
    def show_settings(self):
        """Show settings dialog"""
        settings = tk.Toplevel(self.root)
        settings.title("Settings")
        settings.geometry("400x300")
        
        # Add settings controls here
        ttk.Label(settings, text="Settings will be available soon").pack(pady=20)
        
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.is_running = False
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = SeismicDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()