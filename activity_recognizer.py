# this program recognizes activities
from pathlib import Path 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def load_training_data(base_path="complete-data"):
    data_list = []
    labels = []
    
    path = Path(base_path)
    
    for file_path in path.rglob("*.csv"):
        label = file_path.stem.split("-")[1]
        df = pd.read_csv(file_path)
        data_list.append(df)
        labels.append(label)
    
    print(f"Data loaded. Total files: {len(data_list)}")
    return data_list, labels

def extract_features(df):
    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    
    df_clean = df[sensor_cols]

    df_clean = df_clean.interpolate(method="linear", limit_direction="both")
    df_clean = df_clean.fillna(0)
    
    means = df_clean.mean().values
    stds = df_clean.std().values
    
    fft_magnitudes = np.abs(np.fft.rfft(df_clean.values, axis=0))
    
    fft_magnitudes = fft_magnitudes[1:, :]
    
    fft_means = np.mean(fft_magnitudes, axis=0)
    fft_maxs = np.max(fft_magnitudes, axis=0)

    return np.concatenate([means, stds, fft_means, fft_maxs])

def train_svm(features, labels, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)

    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=15, gamma="scale", probability=False))
    svm_pipeline.fit(X_train, y_train)

    accuracy = svm_pipeline.score(X_test, y_test)
    print(f"SVM Accuracy: {accuracy:.2f}")

    return [svm_pipeline, accuracy]

def train_random_forest(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train, y_train)

    accuracy = rf_model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {accuracy:.2f}\n")
    
    return [rf_model, accuracy]

def get_models():
    data, labels = load_training_data()
    features = [extract_features(df) for df in data]
    
    svm_model, _ = train_svm(features, labels)
    rf_model, _ = train_random_forest(features, labels)
    
    return svm_model, rf_model

def get_prediction(svm_model, rf_model, df):
    features = extract_features(df).reshape(1, -1)
    svm_pred = svm_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    return svm_pred, rf_pred

if __name__ == "__main__":
    data, labels = load_training_data()
    features = [extract_features(df) for df in data]
    
    svm_accs = []
    rf_accs = []
    for i in range(100):
        svm_model, svm_acc = train_svm(features, labels, random_state=i)
        rf_model, rf_acc = train_random_forest(features, labels)
        
        svm_accs.append(svm_acc)
        rf_accs.append(rf_acc)
    print(f"Mean SVM Accuracy: {np.mean(svm_accs):.2f} ± {np.std(svm_accs):.2f}")
    print(f"Mean Random Forest Accuracy: {np.mean(rf_accs):.2f} ± {np.std(rf_accs):.2f}")