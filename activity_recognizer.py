# this program recognizes activities
from pathlib import Path 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

class ActivityRecognizer:
    def __init__(self):
        self.data, self.labels = self.load_training_data()
        self.features = [self.extract_features(df) for df in self.data]
        
        self.svm_model, self.svm_acc = self.train_svm(self.features, self.labels)
        self.rf_model, self.rf_acc = self.train_random_forest(self.features, self.labels)

    def load_training_data(self,base_path="complete-data"):
        data_list = []
        labels = []
        
        path = Path(base_path)
        
        # Look for all CSV files in the directory and subdirectories
        for file_path in path.rglob("*.csv"):
            
            # Extract label
            label = file_path.stem.split("-")[1]
            
            # Read CSV and store data and label
            df = pd.read_csv(file_path)
            data_list.append(df)
            labels.append(label)
        
        print(f"Data loaded. Total files: {len(data_list)}")
        return data_list, labels

    def extract_features(self,df):
        # Drop timestamp
        sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        df_clean = df[sensor_cols]

        # Handle missing values by interpolation, then fill any remaining NaNs with 0
        df_clean = df_clean.interpolate(method="linear", limit_direction="both")
        df_clean = df_clean.fillna(0)
        
        # Calculate mean and std for each sensor axis
        means = df_clean.mean().values
        stds = df_clean.std().values
        
        # Perform FFT and drop DC component
        fft_magnitudes = np.abs(np.fft.rfft(df_clean.values, axis=0))
        fft_magnitudes = fft_magnitudes[1:, :]
        
        # Calculate mean and max of FFT magnitudes for each axis
        fft_means = np.mean(fft_magnitudes, axis=0)
        fft_maxs = np.max(fft_magnitudes, axis=0)

        # Combine all features into a single vector
        return np.concatenate([means, stds, fft_means, fft_maxs])

    def train_svm(self, features, labels, random_state=42):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)

        # Create a pipeline that standardizes the data and then applies SVM
        svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=15, gamma="scale", probability=False))
        svm_pipeline.fit(X_train, y_train)

        # Evaluate accuracy on the test set
        accuracy = svm_pipeline.score(X_test, y_test)

        return [svm_pipeline, accuracy]

    def train_random_forest(self, features, labels):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate accuracy on the test set
        accuracy = rf_model.score(X_test, y_test)
        
        return [rf_model, accuracy]

    def get_predictions(self, df):
        # Extract features
        features = self.extract_features(df).reshape(1, -1)
        
        # Get predictions from both models
        svm_pred = self.svm_model.predict(features)[0]
        rf_pred = self.rf_model.predict(features)[0]
        
        return svm_pred, rf_pred
    
    def print_model_accuracies(self):
        print(f"SVM Accuracy: {self.svm_acc:.2f}")
        print(f"Random Forest Accuracy: {self.rf_acc:.2f}")
        
if __name__ == "__main__":
    recognizer = ActivityRecognizer()
    recognizer.print_model_accuracies()
    
