# this program visualizes activities with pyglet
import activity_recognizer as activity
import pyglet
from activity_recognizer import ActivityRecognizer
from DIPPID import SensorUDP
import time
import threading
from collections import deque
import pandas as pd

PORT = 5700
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class FitnessTrainer(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, caption="ITT Fitness Trainer")
        
        # Initialize Activity Recognizer
        self.activity_recognizer = ActivityRecognizer()
        
        # Data Handling
        self.sensor = SensorUDP(PORT)
        self.data = deque()
        
        # Start sensor reading thread
        self.data_lock = threading.Lock()
        threading.Thread(target=self.sensor_reader, daemon=True).start()
        
        # UI Setup
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.shapes.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, color=(255, 255, 255), batch=self.batch)
        self.title  = pyglet.text.Label("ITT Fitness Trainer", font_size=24, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT-50, anchor_x="center", batch=self.batch)
        self.title.color = (0, 0, 0, 255)
        self.prediction_label = pyglet.text.Label("Waiting for data...", font_size=18, x=WINDOW_WIDTH//2, y=50, anchor_x="center", batch=self.batch)
        self.prediction_label.color = (0, 0, 0, 255)
        
        # Schedule the update function to run every 100ms
        pyglet.clock.schedule_interval(self.update, 1/10)
        
    def sensor_reader(self):
            # Continuously read sensor data and store it in a thread-safe manner
            while True:
                acc = self.sensor.get_value("accelerometer")
                gyro = self.sensor.get_value("gyroscope")
                
                # Only store data if both accelerometer and gyroscope readings are available
                if acc and gyro:
                    sample = {
                        "timestamp": time.time(),
                        "acc_x": acc.get("x", 0),
                        "acc_y": acc.get("y", 0),
                        "acc_z": acc.get("z", 0),
                        "gyro_x": gyro.get("x", 0),
                        "gyro_y": gyro.get("y", 0),
                        "gyro_z": gyro.get("z", 0)
                    }
                
                    # Use timestamp to maintain a rolling window of the last 5 seconds of data
                    now = sample["timestamp"]
                    with self.data_lock:
                        self.data.append(sample) 
                        
                        while self.data and now - self.data[0]["timestamp"] > 5:
                            self.data.popleft()
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.001) 
    
    def sample_data(self):
        # Get a snapshot of the current data in a thread-safe manner
        with self.data_lock:
            if not self.data:
                return None

            records = list(self.data)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp").sort_index()

        # Resample to 100Hz and interpolate missing values
        df_resampled = df.resample("10ms").mean().interpolate(method="linear")

        return df_resampled
            
    def update(self, dt):
        # Get the latest data snapshot and make predictions
        data = self.sample_data()
        svm_pred, rf_pred = self.activity_recognizer.get_predictions(data)
        print(f"SVM Prediction: {svm_pred}, RF Prediction: {rf_pred}")
        
        # Update the prediction label on the UI
        self.prediction_label.text = f"SVM: {svm_pred}, RF: {rf_pred}"
        
    def on_draw(self):
        self.clear()
        self.batch.draw()
        
        
if __name__ == "__main__":
    try:
        game = FitnessTrainer()
        pyglet.app.run()
    except KeyboardInterrupt:
        print("Exiting...")
        
