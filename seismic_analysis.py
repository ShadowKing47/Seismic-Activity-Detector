# %% [markdown]
# # Seismic Data Analysis
# 
# This notebook demonstrates how to load, inspect, and visualize seismic data in .mseed format using ObsPy and Matplotlib.

# %%
# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from pathlib import Path
from datetime import datetime, timedelta

# Set plot style
try:
    plt.style.use('seaborn')
except:
    # Fallback to default style if seaborn is not available
    plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

# %% [markdown]
# ## 1. File Loading and Inspection
# 
# Let's define a function to load and display metadata from .mseed files.

# %%
def inspect_seismic_file(file_path):
    """
    Load and display metadata from a seismic .mseed file.
    
    Args:
        file_path (str): Path to the .mseed file
    """
    try:
        print(f"Loading file: {file_path}")
        stream = read(file_path)
        
        # Print general file information
        print(f"\nFile contains {len(stream)} trace(s)")
        print("=" * 50)
        
        # Print information for each trace
        for i, trace in enumerate(stream):
            print(f"\nTrace {i+1} Information:")
            print("-" * 30)
            stats = trace.stats
            
            # Basic trace information
            print(f"Network: {stats.network}")
            print(f"Station: {stats.station}")
            print(f"Channel: {stats.channel}")
            print(f"Location: {stats.location}")
            print(f"Sampling Rate: {stats.sampling_rate} Hz")
            print(f"Number of Samples: {stats.npts}")
            print(f"Start Time: {stats.starttime}")
            print(f"End Time: {stats.endtime}")
            print(f"Delta: {stats.delta} seconds")
            
            # Data statistics
            print("\nData Statistics:")
            print(f"Min: {np.min(trace.data):.2f}")
            print(f"Max: {np.max(trace.data):.2f}")
            print(f"Mean: {np.mean(trace.data):.2f}")
            print(f"Std Dev: {np.std(trace.data):.2f}")
            
        return stream
            
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

# %% [markdown]
# ## 2. Time Series Visualization
# 
# Let's create a function to plot the seismic waveform.

# %%
def detect_velocity_changes(data, window_size=50, threshold_std=3.0):
    """
    Detect sudden changes in velocity data.
    
    Args:
        data (np.array): Array of velocity values
        window_size (int): Size of the sliding window for analysis
        threshold_std (float): Number of standard deviations to consider as a significant change
        
    Returns:
        list: List of tuples (index, velocity_before, velocity_after, trend)
    """
    changes = []
    if len(data) < window_size * 2:
        return changes
    
    for i in range(window_size, len(data) - window_size):
        # Calculate statistics in the window before and after the current point
        before = data[i-window_size:i]
        after = data[i:i+window_size]
        
        # Calculate mean and standard deviation
        mean_before = np.mean(before)
        std_before = np.std(before)
        mean_after = np.mean(after)
        
        # Check for significant change
        if abs(mean_after - mean_before) > threshold_std * std_before:
            # Determine trend (1 for increasing, -1 for decreasing)
            trend = 1 if mean_after > mean_before else -1
            changes.append((i, mean_before, mean_after, trend))
    
    return changes

def plot_seismic_waveform(stream, arrival_time=None, title=None):
    """
    Plot the seismic waveform from an ObsPy stream with velocity change detection.
    
    Args:
        stream (obspy.Stream): Stream object containing seismic data
        arrival_time (str, optional): Arrival time in ISO format (YYYY-MM-DDTHH:MM:SS.sss)
        title (str, optional): Plot title
    """
    if not stream:
        print("No data to plot")
        return
    
    # Create a figure with two subplots (waveform and gradient)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    for trace in stream:
        # Get data and time axis
        data = trace.data
        times = np.linspace(0, len(data) / trace.stats.sampling_rate, len(data))
        
        # Plot the main trace
        ax1.plot(times, data, 
                label=f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}")
        
        # Detect velocity changes
        changes = detect_velocity_changes(data)
        
        # Plot change points and analyze trends
        for i, (idx, v_before, v_after, trend) in enumerate(changes):
            # Plot vertical line at change point
            change_time = times[idx]
            ax1.axvline(x=change_time, color='red', alpha=0.5, linestyle='--', 
                       label='Velocity Change' if i == 0 else "")
            
            # Add text annotation
            ax1.text(change_time, np.max(data)*0.9, 
                    f"Δv: {v_after-v_before:.2f}\n"
                    f"Trend: {'↑' if trend > 0 else '↓'}",
                    ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
            
            # Print detailed information
            print(f"\nVelocity Change Detected at {change_time:.2f}s:")
            print(f"- Velocity before: {v_before:.4f} m/s")
            print(f"- Velocity after: {v_after:.4f} m/s")
            print(f"- Change: {v_after - v_before:.4f} m/s")
            print(f"- Trend: {'Increasing' if trend > 0 else 'Decreasing'}")
        
        # Add arrival time if provided
        if arrival_time:
            try:
                arrival_dt = datetime.fromisoformat(arrival_time)
                start_dt = trace.stats.starttime.datetime
                arrival_sec = (arrival_dt - start_dt).total_seconds()
                
                if 0 <= arrival_sec <= times[-1]:
                    ax1.axvline(x=arrival_sec, color='green', linestyle='-', 
                              linewidth=2, label='Arrival Time')
            except Exception as e:
                print(f"Error processing arrival time: {str(e)}")
        
        # Plot the velocity gradient (rate of change) in the second subplot
        gradient = np.gradient(data, times)
        ax2.plot(times, gradient, 'b-', alpha=0.6, label='Velocity Gradient')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Mark significant gradient changes
        for idx, _, _, _ in changes:
            ax2.axvline(x=times[idx], color='red', alpha=0.5, linestyle='--')
    
    # Set labels and titles for the plots
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title(title or 'Seismic Waveform with Velocity Changes')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Gradient (m/s²)')
    ax2.set_title('Velocity Gradient')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. Process Multiple Files
# 
# Let's process all .mseed files in a directory.

# %%
def process_directory(directory_path, max_files=5):
    """
    Process all .mseed files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing .mseed files
        max_files (int): Maximum number of files to process (for demo purposes)
    """
    directory = Path(directory_path)
    mseed_files = list(directory.glob('**/*.mseed'))
    
    if not mseed_files:
        print(f"No .mseed files found in {directory_path}")
        return
    
    print(f"Found {len(mseed_files)} .mseed files")
    
    # Process up to max_files
    for i, file_path in enumerate(mseed_files[:max_files]):
        print(f"\n{'='*80}")
        print(f"Processing file {i+1}/{min(len(mseed_files), max_files)}: {file_path.name}")
        print("="*80)
        
        # Inspect the file
        stream = inspect_seismic_file(str(file_path))
        
        # Plot the waveform (using first trace's start time as example arrival time)
        if stream and len(stream) > 0:
            arrival_time = stream[0].stats.starttime.isoformat()
            plot_seismic_waveform(
                stream, 
                arrival_time=arrival_time,
                title=f"Seismic Waveform: {file_path.name}"
            )

# %% [markdown]
# ## 4. Example Usage
# 
# Let's run the analysis on a sample directory.

# %%
if __name__ == "__main__":
    # Define possible data directories to check
    possible_data_dirs = [
        "space_apps_2024_seismic_detection/data",
        "data",
        "space_apps_2024_seismic_detection"
    ]
    
    # Find the first existing data directory
    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
            
    if data_dir is None:
        print("Error: Could not find data directory. Please check the following paths:")
        for dir_path in possible_data_dirs:
            print(f"- {os.path.abspath(dir_path)}")
        exit(1)
    
    if os.path.exists(data_dir):
        print(f"Processing files in: {os.path.abspath(data_dir)}")
        process_directory(data_dir, max_files=2)  # Process first 2 files as example
    else:
        print(f"Directory not found: {data_dir}")
        print("Please update the 'data_dir' variable to point to your data directory.")