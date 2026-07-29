import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import mlflow

# --- SENIOR_TODO: Train on Railway or Locally? ---
# Railway has limited CPU. Train locally and commit the model.
# Or use Railway's volume to store the model.
MODEL_PATH = 'models/'

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    joblib.dump(le, os.path.join(MODEL_PATH, 'type_encoder.joblib'))
    
    X = df.drop(['Target', 'Failure Type'], axis=1)
    y = df['Target']
    return X, y, le

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.joblib'))
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    print(f"Test F1 Score: {f1:.4f}")
    
    joblib.dump(model, os.path.join(MODEL_PATH, 'model.joblib'))
    return model

if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    X, y, _ = load_and_preprocess('predictive_maintenance.csv')
    train_model(X, y)
    print("✅ Training complete! Models saved to 'models/'")