# this program gathers sensor data
import time
import sys
import pandas as pd
from DIPPID import SensorUDP

# Configuration
PORT = 5700
DURATION = 10.0      # 10 seconds of recording

# Global state
is_recording = False

def handle_button_press(data):
    global is_recording
    if int(data) == 1 and not is_recording:
        print("\n[!] Button pressed! Starting 10-second capture...")
        is_recording = True

def main():
    global is_recording
    
    print("--- DIPPID Data Gathering ---")
    user_name = input("Enter your name: ").strip()
    activity = input("Enter activity (running, rowing, lifting, jumpingjacks): ").strip().lower()
    trial_num = input("Enter trial number: ").strip()
    
    file_name = f"{user_name}-{activity}-{trial_num}.csv"

    sensor = SensorUDP(PORT)
    sensor.register_callback("button_1", handle_button_press)

    print(f"\nReady to record: {file_name}")
    print("Press 'Button 1' on your DIPPID device to start.")

    while not is_recording:
        time.sleep(0.1)
        
    # Capture data as quickly as possible for 10 seconds.
    data_log = []
    start_time = time.time()
    
    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= DURATION:
            break
            
        acc = sensor.get_value("accelerometer")
        gyro = sensor.get_value("gyroscope")
        
        if acc is not None and gyro is not None:
            data_log.append({
                "timestamp": elapsed_time,
                "acc_x": acc.get("x", 0),
                "acc_y": acc.get("y", 0),
                "acc_z": acc.get("z", 0),
                "gyro_x": gyro.get("x", 0),
                "gyro_y": gyro.get("y", 0),
                "gyro_z": gyro.get("z", 0)
            })
            
        # Short pause to keep the loop from hogging the CPU.
        time.sleep(0.001) 

    print("\n[!] 10 seconds reached. Stopping capture.")
    sensor.disconnect()
    
    # Resample to 100Hz.
    if len(data_log) > 0:
        print("Resampling data to 100Hz...")
        df = pd.DataFrame(data_log)
        
        # Convert timestamps so Pandas can work with them.
        df["timestamp"] = pd.to_timedelta(df["timestamp"], unit="s")
        
        # Use timestamp as the index for resampling.
        df = df.set_index("timestamp")
        
        # "10ms" = 100Hz. Mean smooths extra samples, interpolation fills gaps.
        df_resampled = df.resample("10ms").mean().interpolate(method="linear")
        
        # Put the data back into the required format.
        df_resampled = df_resampled.reset_index()
        
        # Convert timestamps back to seconds.
        df_resampled["timestamp"] = df_resampled["timestamp"].dt.total_seconds()
        
        # Recreate the ID column.
        df_resampled["id"] = range(len(df_resampled))
        
        # Keep the expected column order.
        columns = ["id", "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        df_resampled = df_resampled[columns]
        
        # Trim any extra rows past 10 seconds.
        df_resampled = df_resampled.head(1000)
        
        # Save the file.
        df_resampled.to_csv("data/" + file_name, index=False)
        print(f"Success! Saved exactly {len(df_resampled)} resampled rows to '{file_name}'.")
    else:
        print("Error: No data was captured.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
        sys.exit(0)