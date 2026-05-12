# this program visualizes activities with pyglet

import activity_recognizer as activity
import pyglet
from activity_recognizer import get_models, extract_features, get_prediction
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
        self.svm_model, self.rf_model = get_models()
        
        # Data Handling
        self.sensor = SensorUDP(PORT)
        self.data = deque()
        self.data_lock = threading.Lock()
        threading.Thread(target=self.sensor_reader, daemon=True).start()
        
        # UI Setup
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.shapes.Rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, color=(255, 255, 255), batch=self.batch)
        self.title  = pyglet.text.Label("ITT Fitness Trainer", font_size=24, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT-50, anchor_x="center", batch=self.batch)
        
        pyglet.clock.schedule_interval(self.update, 1)
        
    def sensor_reader(self):
            while True:
                acc = self.sensor.get_value("accelerometer")
                gyro = self.sensor.get_value("gyroscope")
                
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
                
                    now = sample["timestamp"]
                    with self.data_lock:
                        self.data.append(sample) 
                        
                        while self.data and now - self.data[0]["timestamp"] > 10:
                            self.data.popleft()
                    
                time.sleep(0.001) 
    
    def sample_data(self):
        with self.data_lock:
            if not self.data:
                return None

            records = list(self.data)

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp").sort_index()

        df_resampled = df.resample("10ms").mean().interpolate(method="linear")
        print(df_resampled.head())
        print(df_resampled.shape)
        return df_resampled
            
    def update(self, dt):
        data = self.sample_data()
        svm_pred, rf_pred = get_prediction(self.svm_model, self.rf_model, data)
        print(f"SVM Prediction: {svm_pred}, RF Prediction: {rf_pred}")
        
             
    def on_draw(self):
        self.clear()
        self.batch.draw()
        
        
if __name__ == "__main__":
    try:
        game = FitnessTrainer()
        pyglet.app.run()
    except KeyboardInterrupt:
        print("Exiting...")
        
