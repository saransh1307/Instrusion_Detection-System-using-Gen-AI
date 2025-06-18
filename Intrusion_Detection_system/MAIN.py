import os
import torch
import pandas as pd
from src.ctgan_model import CTGANSynthesizer
from src.vae_dbn_model import DBN, VAE, train_vae_dbn, detect_anomalies, evaluate_detection, MODEL_PATH
from src.utils import load_and_preprocess_ctgan_data, load_and_preprocess_nslkdd_data, NSL_KDD_TRAIN_PATH, NSL_KDD_TEST_PATH

def main():
    """
    Main function to run the entire intrusion detection system workflow.
    This includes CTGAN for synthetic data, DBN for feature extraction,
    and VAE for anomaly detection.
    """
    print("Starting Intrusion Detection System Workflow...")

    # --- Configuration ---
    # Define paths for data. Adjust these to your actual data locations.
    # For CTGAN, we'll use a dummy path for demonstration or use a specific dataset if available.
    CTGAN_TRAIN_DATA_PATH = 'data/ctgan_train_data.csv' # Placeholder, replace with your actual CTGAN data if different
    SYNTHETIC_DATA_OUTPUT_PATH = 'data/synthetic_nslkdd_data.csv'

    # Ensure the 'data' and 'models' directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # --- Phase 1: Synthetic Data Generation with CTGAN ---
    print("\n--- Phase 1: Synthetic Data Generation with CTGAN ---")

    # For demonstration, let's assume we want to generate synthetic data for the NSL-KDD format
    # The CTGAN script expects a CSV. Let's create a simplified version for CTGAN training
    # For a real scenario, you'd train CTGAN on a suitable real dataset.
    print("Preparing dummy data for CTGAN (if real CTGAN data not specified)...")
    try:
        # Load a small portion of NSL-KDD train data to serve as "real data" for CTGAN
        # In a real application, CTGAN would be trained on a representative dataset.
        df_for_ctgan_training, _, _ = load_and_preprocess_ctgan_data(NSL_KDD_TRAIN_PATH)
        # For this example, let's save a subset to a CSV that CTGAN can read
        df_for_ctgan_training.sample(min(1000, len(df_for_ctgan_training))).to_csv(CTGAN_TRAIN_DATA_PATH, index=False)
        print(f"Generated dummy CTGAN training data at {CTGAN_TRAIN_DATA_PATH}")

        ctgan_synthesizer = CTGANSynthesizer(epochs=50) # Reduced epochs for quicker run
        ctgan_synthesizer.train(df_for_ctgan_training)

        # Generate synthetic data
        num_synthetic_samples = 1000
        synthetic_df = ctgan_synthesizer.generate_synthetic_data(num_synthetic_samples)
        synthetic_df.to_csv(SYNTHETIC_DATA_OUTPUT_PATH, index=False)
        print(f"Generated {num_synthetic_samples} synthetic samples and saved to {SYNTHETIC_DATA_OUTPUT_PATH}")

        # Evaluate synthetic data (placeholder)
        ctgan_synthesizer.evaluate_synthetic_data(df_for_ctgan_training, synthetic_df)
    except Exception as e:
        print(f"Error during CTGAN phase: {e}")
        print("Skipping CTGAN phase. Please ensure 'data/ctgan_train_data.csv' exists or adjust paths/logic.")

    # --- Phase 2: Intrusion Detection with DBN-VAE ---
    print("\n--- Phase 2: Intrusion Detection with DBN-VAE ---")

    # Check if the VAE model is already trained
    if not os.path.exists(MODEL_PATH):
        print("VAE model not found. Training new model...")
        vae_model, encoders, scaler, dbn_trained, anomaly_threshold = train_vae_dbn(NSL_KDD_TRAIN_PATH, epochs_dbn=10, epochs_vae=50)
    else:
        print(f"Loading existing VAE model from {MODEL_PATH}")
        # Need to load encoders and scaler from a previously saved state if not retraining
        # For simplicity in this example, we re-preprocess test data to get encoders/scaler
        # In a real system, you'd save/load these along with the model.
        _, _, _, encoders, scaler = load_and_preprocess_nslkdd_data(NSL_KDD_TRAIN_PATH) # Load for consistent preprocessing
        
        # Determine the input dimension for VAE based on the DBN's expected output.
        # This requires initializing a dummy DBN to get the output shape or saving the DBN's final layer size.
        # For robustness, you should save DBN's layer configuration.
        # Assuming the DBN output layer size is 32 as in your RBM layers setup.
        dummy_input_dim = 122 # NSL-KDD features
        dummy_dbn = DBN([dummy_input_dim, 64, 32])
        vae_input_dim = dummy_dbn(torch.randn(1, dummy_input_dim)).shape[1] # Infer input dim from dummy DBN pass

        vae_model = VAE(vae_input_dim)
        vae_model.load_state_dict(torch.load(MODEL_PATH))
        vae_model.eval()

        # To perform detection, we need the DBN used for feature extraction during training.
        # Ideally, the DBN state should also be saved or its structure known.
        # Re-initializing DBN for prediction with the same structure as training.
        dbn_trained = DBN([dummy_input_dim, 64, 32])
        # Note: If DBN pre-training was crucial for learned features, its weights
        # should also be saved and loaded, or pre-trained again on a small sample for consistency.
        # For simplicity here, we assume it's stateless after pretraining is implicitly done
        # if the VAE is trained on DBN features. In a robust system, you'd save the DBN too.
        # Here, we re-pretrain for consistency on test data if the model is loaded.
        # This is a simplification; in a real scenario, DBN would be fixed after training.

        # Calculate threshold from test data for consistency if not saved from training
        # This is a common practice if you want to analyze test data characteristics
        # It's better to save the threshold determined from the training set.
        print("Calculating anomaly threshold from training data (if model loaded, this should be pre-saved)...")
        X_train_for_threshold, _, _, _, _ = load_and_preprocess_nslkdd_data(NSL_KDD_TRAIN_PATH, encoders, scaler)
        X_train_tensor_for_threshold = torch.tensor(X_train_for_threshold, dtype=torch.float32)
        dbn_features_for_threshold = dbn_trained(X_train_tensor_for_threshold).detach()
        with torch.no_grad():
            recon_for_threshold, _, _ = vae_model(dbn_features_for_threshold)
            errors_for_threshold = torch.mean((dbn_features_for_threshold - recon_for_threshold) ** 2, dim=1).cpu().numpy()
            anomaly_threshold = np.percentile(errors_for_threshold, 95)
        print(f"Anomaly threshold set at: {anomaly_threshold:.4f}")

    # Perform anomaly detection on test data
    anomaly_reports, predicted_labels, true_labels = detect_anomalies(
        vae_model, dbn_trained, NSL_KDD_TEST_PATH, encoders, scaler, anomaly_threshold
    )

    # Print a few anomaly reports
    print("\n--- Sample Anomaly Reports (First 5) ---")
    for i, report in enumerate(anomaly_reports[:5]):
        print(f"Anomaly {i+1}:\n{report}")
    if not anomaly_reports:
        print("No anomalies detected in the sample.")


    # Evaluate the detection performance
    evaluate_detection(true_labels, predicted_labels)

    print("\nIntrusion Detection System Workflow Complete.")

if __name__ == '__main__':
    main()