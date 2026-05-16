import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib

# The 43 column names specific to NSL-KDD 
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", 
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"
]

def load_data(filepath):
    df = pd.read_csv(filepath, names=COLUMN_NAMES, header=None)
    
    # Preprocess binary target: mapping any attack to 1 ('anomaly'), 'normal' to 0
    df['class'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # We drop attack_type and difficulty_level now that we created our binary target 'class'
    df = df.drop(['attack_type', 'difficulty_level'], axis=1)
    return df

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'KDDTrain+.txt')
    models_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please run data/download_dataset.py first.")
        return

    print("Loading NSL-KDD dataset...")
    df = load_data(data_path)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 41 remaining features
    categorical_features = ['protocol_type', 'service', 'flag']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # We add SMOTE to exactly mirror the architecture of the reference project
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Starting MLflow run...")
    mlflow.set_experiment("NSL_KDD_Intrusion_Detection")
    
    with mlflow.start_run():
        print("Training Random Forest model (this may take a minute depending on hardware)...")
        model_pipeline.fit(X_train, y_train)
        
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 15)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        
        mlflow.sklearn.log_model(model_pipeline, "model")
        
        model_path = os.path.join(models_dir, 'intrusion_detection_model.pkl')
        joblib.dump(model_pipeline, model_path)
        print(f"Model saved locally to {model_path}")

if __name__ == "__main__":
    train_model()
