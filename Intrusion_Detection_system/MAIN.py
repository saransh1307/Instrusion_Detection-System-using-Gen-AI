import os
import torch
import pandas as pd
import numpy as np
from Intrusion_Detection_system.Models.Synthethic_Data_generation import CTGANSynthesizer  # Importing CTGANSynthesizer for synthetic data generation
from Intrusion_Detection_system.Models.VAE_DBN_model import (
    DBN, VAE, train_vae_dbn, detect_anomalies, evaluate_detection,
    MODEL_PATH
)
from Intrusion_Detection_system.utils import (
    load_and_preprocess_ctgan_data,
    load_and_preprocess_nslkdd_data,
    NSL_KDD_TRAIN_PATH,
    NSL_KDD_TEST_PATH
)

# -------------------- PATH CONFIGURATION --------------------
CTGAN_TRAIN_DATA_PATH = "Intrusion_Detection_system/DATA/RAW/KDDTrain+.txt"
SYNTHETIC_DATA_OUTPUT_PATH = "Intrusion_Detection_system/DATA/Synthetic/synthetic_data.csv"
MODEL_PATH = "Intrusion_Detection_system/models/vae_model_with_dbn.pth"
# THRESHOLD_PATH = "Intrusion_Detection_system/models/anomaly_threshold.txt"
DBN_MODEL_PATH = "Intrusion_Detection_system/models/dbn_weights.pth"
THRESHOLD_PATH = "Intrusion_Detection_system/models/anomaly_threshold.npy"



def main():
    print("\n🚀 Starting Intrusion Detection System Workflow...")

    # Ensure required directories exist
    os.makedirs("Intrusion_Detection_system/DATA/Synthetic/synthetic_data.csv", exist_ok=True)
    os.makedirs("Intrusion_Detection_system/models", exist_ok=True)

    # -------------------- PHASE 1: CTGAN --------------------
    print("\n--- Phase 1: Synthetic Data Generation with CTGAN ---")

    # Load and preprocess original training data for CTGAN
    df_for_ctgan_training, _, _ = load_and_preprocess_ctgan_data(CTGAN_TRAIN_DATA_PATH)  # we are using NSLKDD dataset for training the CTGAN model. The CTGAN model is trained on the NSL-KDD dataset to generate synthetic data that resembles the original data distribution.

    # Train CTGAN
    ctgan_synthesizer = CTGANSynthesizer(epochs=50)   # ctgan_synthesizer came from the Synthethic_Data_generation.py file. The CTGANSynthesizer class is used to train a CTGAN model on the NSL-KDD dataset.
    ctgan_synthesizer.train(df_for_ctgan_training) 

    # Generate synthetic samples
    synthetic_df = ctgan_synthesizer.generate_synthetic_data(num_samples=1000) # num_samples=1000 means we are generating 1000 synthetic samples. The number of samples can be changed as per the requirement.
    synthetic_df.to_csv(SYNTHETIC_DATA_OUTPUT_PATH, index=False) # Save synthetic data to CSV because Pandas DataFrame is used to store the synthetic data.
    print(f"✅ Saved synthetic data to: {SYNTHETIC_DATA_OUTPUT_PATH}")  # The synthetic data is saved to the specified path.

    # Optional: evaluate synthetic quality
    # ctgan_synthesizer.evaluate_synthetic_data(df_for_ctgan_training, synthetic_df)

    # -------------------- PHASE 2: VAE + DBN --------------------
    print("\n--- Phase 2: Intrusion Detection with DBN-VAE ---")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(DBN_MODEL_PATH) and os.path.exists(THRESHOLD_PATH)):
        print("Model or artifacts not found. Training VAE+DBN and computing threshold...")
        vae_model, encoders, scaler, dbn_trained, anomaly_threshold = train_vae_dbn(NSL_KDD_TRAIN_PATH)
    else:
        print("📦 Loading saved model and preprocessing objects...")

        # Load encoders/scaler to ensure consistent preprocessing
        _, _, _, encoders, scaler = load_and_preprocess_nslkdd_data(NSL_KDD_TRAIN_PATH)

        # Initialize and load DBN
        dummy_input_dim = 122
        dbn_trained = DBN([dummy_input_dim, 64, 32])
        dbn_trained.load_state_dict(torch.load(DBN_MODEL_PATH))
        dbn_trained.eval()

        # Initialize and load VAE
        vae_input_dim = dbn_trained(torch.randn(1, dummy_input_dim)).shape[1]
        vae_model = VAE(vae_input_dim)
        vae_model.load_state_dict(torch.load(MODEL_PATH))
        vae_model.eval()

        # Load threshold
        anomaly_threshold = float(np.load(THRESHOLD_PATH))
        print(f"📈 Loaded anomaly threshold: {anomaly_threshold:.4f}")

    # -------------------- PHASE 3: Anomaly Detection --------------------
    print("\n--- Phase 3: Anomaly Detection ---")

    anomaly_reports, predicted_labels, true_labels = detect_anomalies(
        vae_model, dbn_trained, NSL_KDD_TEST_PATH, encoders, scaler, anomaly_threshold
    )

    print("\n--- Sample Anomaly Reports (Top 5) ---")
    for i, report in enumerate(anomaly_reports[:5]):
        print(f"\n🔍 Anomaly {i+1}:\n{report}")

    if not anomaly_reports:
        print("✅ No anomalies detected in the sample.")

    # -------------------- Evaluation --------------------
    evaluate_detection(true_labels, predicted_labels)
    print("\n✅ Intrusion Detection System Workflow Complete.")

if __name__ == '__main__':
    main()
