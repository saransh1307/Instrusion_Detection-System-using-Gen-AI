import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Intrusion_Detection_system.Models.ctgan import CTGAN
import os

# Load and preprocess data
def load_data(file_path):
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

# Train CTGAN
def train_ctgan(df, epochs=300):
    ctgan = CTGAN(epochs=epochs)
    ctgan.fit(df)
    return ctgan

# Generate synthetic data
def generate_synthetic_data(ctgan, num_samples):
    synthetic_data = ctgan.sample(num_samples)
    return synthetic_data

# Evaluate synthetic data
def evaluate_synthetic_data(real_data, synthetic_data):
    # Placeholder for evaluation logic
    # You can implement statistical tests or train a classifier to distinguish between real and synthetic data
    pass

def main():
    # Paths to your data files
    train_file = 'path_to_train_data.csv'
    test_file = 'path_to_test_data.csv'

    # Load and preprocess data
    train_df, encoders, scaler = load_data(train_file)
    test_df, _, _ = load_data(test_file)

    # Train CTGAN
    ctgan = train_ctgan(train_df)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(ctgan, num_samples=1000)

    # Evaluate synthetic data
    evaluate_synthetic_data(train_df, synthetic_data)

    # Save synthetic data
    synthetic_data.to_csv('synthetic_data.csv', index=False)

if __name__ == '__main__':
    main()
