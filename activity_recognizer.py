# this program recognizes activities
from DIPPID import SensorUDP
import time
import pandas as pd
from pathlib import Path
from threading import Thread

NAME = "jonas"

PORT = 5700
SAMPLING_RATE = 1
SAMPLING_INTERVAL = 1 / SAMPLING_RATE
CURRENTLY_CAPTURING = False

sensor = SensorUDP(PORT)

def handler(data, activity):
    if data == 1:
        global CURRENTLY_CAPTURING
        if CURRENTLY_CAPTURING:
            print("Already capturing data. Please wait until the current capture is finished.")
            return
        print(f"Capturing {activity} data...")
        Thread(target=capture_data, args=(NAME, activity), daemon=True).start()
        CURRENTLY_CAPTURING = True

# Register callbacks for button presses
sensor.register_callback("button_1", lambda data: handler(data, "running"))
sensor.register_callback("button_2", lambda data: handler(data, "rowing"))
sensor.register_callback("button_3", lambda data: handler(data, "jumping_jacks"))
sensor.register_callback("button_4", lambda data: handler(data, "lifting"))

def capture_data(name, activity):
    global CURRENTLY_CAPTURING
    duration = 10.0
    t0 = time.perf_counter()
    rows = []

    while time.perf_counter() - t0 < duration:
        iter_start = time.perf_counter()

        accelerometer_data = sensor.get_value("accelerometer")
        gyroscope_data = sensor.get_value("gyroscope")

        if accelerometer_data and gyroscope_data:
            rows.append({
                "id": len(rows),
                "timestamp": time.time(),
                "acc_x": accelerometer_data["x"],
                "acc_y": accelerometer_data["y"],
                "acc_z": accelerometer_data["z"],
                "gyro_x": gyroscope_data["x"],
                "gyro_y": gyroscope_data["y"],
                "gyro_z": gyroscope_data["z"]
            })

        elapsed = time.perf_counter() - iter_start
        sleep_time = SAMPLING_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Finished capturing {activity} data. Saving to CSV...")

    working_directory = Path("assignment-03-activity-recognition-taylan-jonas") / "data"
    working_directory.mkdir(parents=True, exist_ok=True)

    base_name = f"{name}-{activity}.csv"
    target = working_directory / base_name
    suffix = 1
    while target.exists():
        target = working_directory / f"{name}-{activity}-{suffix}.csv"
        suffix += 1

    pd.DataFrame(rows).to_csv(target, index=False)
    CURRENTLY_CAPTURING = False

try:
    while True:
        time.sleep(1)                
except KeyboardInterrupt:
    print("Exiting...")
    sensor.stop()