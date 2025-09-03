import os
from obspy import read

def test_data_loading(data_dir):
    print("Testing data loading...")
    mseed_files = [f for f in os.listdir(data_dir) if f.endswith('.mseed')]
    
    if not mseed_files:
        print(f"Error: No .mseed files found in {data_dir}")
        return False
    
    print(f"Found {len(mseed_files)} .mseed files")
    
    # Try to read the first file
    test_file = os.path.join(data_dir, mseed_files[0])
    print(f"\nReading file: {test_file}")
    
    try:
        st = read(test_file)
        print("Successfully read data!")
        print(f"Number of traces: {len(st)}")
        if len(st) > 0:
            print("\nFirst trace info:")
            print(st[0])
            print("\nStats:")
            print(st[0].stats)
            print("\nData shape:", st[0].data.shape)
            print("Sampling rate:", st[0].stats.sampling_rate, "Hz")
        return True
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return False

if __name__ == "__main__":
    data_dir = r"C:\Users\91995\OneDrive\Desktop\Nasa Space App hackathon\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA"
    test_data_loading(data_dir)
