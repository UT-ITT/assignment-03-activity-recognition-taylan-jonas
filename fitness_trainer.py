import pyglet
from pyglet import shapes
from pyglet.window import key
import threading
import time
from collections import deque
import pandas as pd

# Import from your modules
from activity_recognizer import ActivityRecognizer
from DIPPID import SensorUDP

PORT = 5700
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class FitnessTrainer(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, caption="ITT Fitness Trainer", resizable=True)
        
        print("Loading data and training models... This might take a moment.")
        self.activity_recognizer = ActivityRecognizer()
        
        self.sensor = SensorUDP(PORT)
        self.data = deque()
        self.data_lock = threading.Lock()
        self.min_samples_required = 20 
        
        threading.Thread(target=self.sensor_reader, daemon=True).start()
        
        # --- Game Logic Variables ---
        self.state = "MENU"
        self.target_activity = None
        self.current_set = 1
        self.max_sets = 3
        self.time_per_set = 30.0 
        self.active_time_remaining = self.time_per_set
        self.break_duration = 30.0 
        self.break_time_remaining = 0.0
        
        # --- UI Setup ---
        self.batch = pyglet.graphics.Batch()
        
        self.background = shapes.Rectangle(0, 0, self.width, self.height, color=(30, 30, 46), batch=self.batch)
        self.header_bg = shapes.Rectangle(0, self.height - 80, self.width, 80, color=(24, 24, 37), batch=self.batch)
        
        # Made the box taller to allow for more text spacing
        self.status_box = shapes.Rectangle(0, 0, 600, 500, color=(49, 50, 68), batch=self.batch) 
        
        self.title = pyglet.text.Label("AI Fitness Trainer", font_name='Arial', font_size=28,
                                       anchor_x="center", anchor_y="center", color=(205, 214, 244, 255), batch=self.batch)
        
        self.instruction_label = pyglet.text.Label("", font_name='Arial', font_size=18,
                                                   anchor_x="center", anchor_y="center", multiline=True, width=500,
                                                   color=(186, 194, 222, 255), batch=self.batch)
        
        self.activity_label = pyglet.text.Label("Loading...", font_name='Arial', font_size=32,
                                                anchor_x="center", anchor_y="center", color=(166, 227, 161, 255), batch=self.batch)
        
        self.models_label = pyglet.text.Label("", font_name='Arial', font_size=14,
                                              anchor_x="center", anchor_y="center", color=(186, 194, 222, 255), batch=self.batch)
        
        self.bar_bg = shapes.Rectangle(0, 0, 400, 20, color=(24, 24, 37), batch=self.batch)
        self.bar_fill = shapes.Rectangle(0, 0, 0, 20, color=(166, 227, 161), batch=self.batch)
        
        # --- Animation & Image Setup ---
        self.animations = {
            "jumpingjack": self.load_animation("jumpingjack"),
            "lifting": self.load_animation("lifting"),
            "rowing": self.load_animation("rowing"),
            "running": self.load_animation("running")
        }
        
        # Changed to ./img/
        try:
            vic_img = pyglet.image.load("./img/victory.png")
            vic_img.anchor_x = vic_img.width // 2
            vic_img.anchor_y = vic_img.height // 2
            self.animations["victory"] = vic_img
        except Exception:
            print("Warning: victory.png not found in ./img/ folder.")
            self.animations["victory"] = pyglet.image.create(1, 1)

        self.sprite = pyglet.sprite.Sprite(img=self.animations["running"] or pyglet.image.create(1,1), batch=self.batch)
        
        # Made the sprite smaller (0.35 instead of 0.5)
        self.sprite.scale = 0.35 
        self.sprite.visible = False
        
        self.on_resize(self.width, self.height)
        pyglet.clock.schedule_interval(self.update, 1/10)
        self.set_menu_state()

    # --- Window & Input Handling ---
    
    def on_resize(self, width, height):
        """Calculates beautifully spaced coordinates so nothing overlaps or clusters."""
        super().on_resize(width, height)
        
        self.background.width = width
        self.background.height = height
        self.header_bg.width = width
        self.header_bg.y = height - 80
        self.title.x = width // 2
        self.title.y = height - 40
        
        self.status_box.x = width // 2 - 300
        self.status_box.y = height // 2 - 250
        
        # --- Increased Y-Spacing to prevent clutter ---
        self.sprite.x = width // 2
        self.sprite.y = height // 2 + 100        # Moved images slightly higher
        
        self.bar_bg.x = width // 2 - 200
        self.bar_bg.y = height // 2 - 10         # Progress bar directly below image
        self.bar_fill.x = width // 2 - 200
        self.bar_fill.y = height // 2 - 10
        
        self.activity_label.x = width // 2
        self.activity_label.y = height // 2 - 70 # Big text lower down
        
        self.instruction_label.x = width // 2
        self.instruction_label.y = height // 2 - 140 # Instructions nicely spaced below big text
        
        self.models_label.x = width // 2
        self.models_label.y = height // 2 - 220  # Sensor status tucked at the very bottom

    def on_key_press(self, symbol, modifiers):
        if symbol == key.F11:
            self.set_fullscreen(not self.fullscreen)
            
        if self.state == "MENU":
            if symbol == key._1: self.start_workout("jumpingjack")
            elif symbol == key._2: self.start_workout("lifting")
            elif symbol == key._3: self.start_workout("rowing")
            elif symbol == key._4: self.start_workout("running")
            
        elif self.state == "FINISHED":
            if symbol == key.SPACE:
                self.set_menu_state()

    # --- Game Logic States ---

    def set_menu_state(self):
        self.state = "MENU"
        self.sprite.visible = False
        self.bar_bg.visible = False
        self.bar_fill.visible = False
        self.activity_label.text = ""
        self.activity_label.color = (205, 214, 244, 255)
        self.instruction_label.text = (
            "Select an exercise to begin your workout:\n\n"
            "Press [1] - Jumping Jacks\n"
            "Press [2] - Lifting\n"
            "Press [3] - Rowing\n"
            "Press [4] - Running"
        )
        
    def start_workout(self, activity):
        self.target_activity = activity
        self.current_set = 1
        self.state = "EXERCISE"
        self.active_time_remaining = self.time_per_set
        
        self.bar_bg.visible = True
        self.bar_fill.visible = True

    # --- Background Data Processing ---

    def load_animation(self, name):
        try:
            # Changed to ./img/
            img1 = pyglet.image.load(f"./img/{name}_1.png")
            img2 = pyglet.image.load(f"./img/{name}_2.png")
            img1.anchor_x, img1.anchor_y = img1.width // 2, img1.height // 2
            img2.anchor_x, img2.anchor_y = img2.width // 2, img2.height // 2
            return pyglet.image.Animation.from_image_sequence([img1, img2], duration=0.4)
        except Exception:
            return None
        
    def sensor_reader(self):
        while True:
            acc = self.sensor.get_value("accelerometer")
            gyro = self.sensor.get_value("gyroscope")
            
            if acc and gyro:
                sample = {
                    "timestamp": time.time(),
                    "acc_x": acc.get("x", 0.0), "acc_y": acc.get("y", 0.0), "acc_z": acc.get("z", 0.0),
                    "gyro_x": gyro.get("x", 0.0), "gyro_y": gyro.get("y", 0.0), "gyro_z": gyro.get("z", 0.0)
                }
                now = sample["timestamp"]
                with self.data_lock:
                    self.data.append(sample) 
                    while self.data and now - self.data[0]["timestamp"] > 5:
                        self.data.popleft()
            time.sleep(0.01) 
    
    def sample_data(self):
        with self.data_lock:
            if len(self.data) < self.min_samples_required:
                return None
            records = list(self.data)

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp").sort_index()

        try:
            df_resampled = df.resample("10ms").mean().interpolate(method="linear").dropna()
            if len(df_resampled) > 10:
                return df_resampled
        except Exception:
            pass
        return None

    def update_visuals(self, target):
        target_lower = str(target).lower()
        matched_anim = None
        for key in self.animations:
            if key in target_lower: 
                matched_anim = self.animations[key]
                break
                
        if matched_anim:
            if self.sprite.image != matched_anim:
                self.sprite.image = matched_anim
            self.sprite.visible = True
        else:
            self.sprite.visible = False 

    # --- Main Loop (Game Logic + Predictions) ---

    def update(self, dt):
        data = self.sample_data()
        rf_pred = "waiting"
        
        if data is not None:
            try:
                svm_pred, rf_pred = self.activity_recognizer.get_predictions(data)
                self.models_label.text = f"Sensor currently detects: {rf_pred.upper()}"
            except Exception:
                pass

        if self.state == "MENU":
            return 
            
        elif self.state == "EXERCISE":
            self.instruction_label.text = f"Target: {self.target_activity.upper()} | Set {self.current_set} of {self.max_sets}"
            self.update_visuals(self.target_activity) 
            
            if self.target_activity in str(rf_pred).lower():
                self.active_time_remaining -= dt
                self.activity_label.text = f"{self.active_time_remaining:.1f}s remaining!"
                self.activity_label.color = (166, 227, 161, 255) 
            else:
                self.activity_label.text = "Perform correct movement!"
                self.activity_label.color = (243, 139, 168, 255) 
                
            progress_ratio = max(0, 1.0 - (self.active_time_remaining / self.time_per_set))
            self.bar_fill.width = 400 * progress_ratio
            
            if self.active_time_remaining <= 0:
                if self.current_set >= self.max_sets:
                    self.state = "FINISHED"
                else:
                    self.current_set += 1
                    self.break_time_remaining = self.break_duration
                    self.state = "BREAK"
                    
        elif self.state == "BREAK":
            self.break_time_remaining -= dt
            self.bar_bg.visible = False
            self.bar_fill.visible = False
            self.sprite.visible = False
            
            self.instruction_label.text = "Catch your breath!"
            self.activity_label.text = f"BREAK: {self.break_time_remaining:.1f}s"
            self.activity_label.color = (137, 180, 250, 255) 
            
            if self.break_time_remaining <= 0:
                self.active_time_remaining = self.time_per_set
                self.bar_bg.visible = True
                self.bar_fill.visible = True
                self.state = "EXERCISE"
                
        elif self.state == "FINISHED":
            self.bar_bg.visible = False
            self.bar_fill.visible = False
            
            self.update_visuals("victory")
            
            self.activity_label.text = "WORKOUT COMPLETE!"
            self.activity_label.color = (249, 226, 175, 255) 
            self.instruction_label.text = "Excellent job! Press [SPACE] to return to the Main Menu."

    def on_draw(self):
        self.clear()
        self.batch.draw()

if __name__ == "__main__":
    try:
        app = FitnessTrainer()
        pyglet.app.run()
    except KeyboardInterrupt:
        print("\nExiting Fitness Trainer...")