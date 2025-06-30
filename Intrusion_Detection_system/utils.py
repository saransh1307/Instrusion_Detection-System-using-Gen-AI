import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tempfile
import numpy as np
import torch

# Constants - it's good practice to define these globally or pass them
# If you have fixed paths for datasets, define them here or in a config file
NSL_KDD_TRAIN_PATH = "DATA/RAW/KDDTrain+_20Percent.txt"
NSL_KDD_TEST_PATH = "DATA/RAW/KDDTest+.txt"

FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

def load_and_preprocess_ctgan_data(file_path):
    """
    Loads and preprocesses data specifically for CTGAN.
    Handles categorical and numerical scaling.
    """
    df = pd.read_csv(file_path)
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    # Scale numerical variables
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df, encoders, scaler

def load_and_preprocess_nslkdd_data(filepath, encoders=None, scaler=None):
    df = pd.read_csv(filepath, names=FEATURE_NAMES)

    # Rename 'class' column to 'attack_type' for clarity
    if 'class' in df.columns:
        df.rename(columns={'class': 'attack_type'}, inplace=True)

    # Drop 'difficulty' column if present
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])

    original_df = df.copy()

    # Separate features and target
    X = df.drop(columns=['attack_type'])
    y = df['attack_type']

    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            # Handle unseen labels by mapping them to 'unknown'
            known_classes = set(encoders[col].classes_)
            X[col] = X[col].apply(lambda val: val if val in known_classes else 'unknown')

            # If 'unknown' is not already in the encoder classes, add it
            if 'unknown' not in encoders[col].classes_:
                encoders[col].classes_ = np.append(encoders[col].classes_, 'unknown')

            X[col] = encoders[col].transform(X[col])

    # Scale numerical features
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Convert attack_type to binary (0 for normal, 1 for malicious)
    y_binary = y.apply(lambda x: 0 if x == 'normal' else 1)

    return X_scaled, y_binary, original_df, encoders, scaler

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Calculates the VAE loss, which is a combination of
    reconstruction loss (MSE) and KL divergence.
    """
    recon_loss = torch.nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss