import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import load_and_preprocess_nslkdd_data, vae_loss_function, NSL_KDD_TRAIN_PATH, NSL_KDD_TEST_PATH

# Constants specific to VAE/DBN model
MODEL_PATH = "models/vae_model_with_dbn.pth"

# ------------------------------ DBN ------------------------------
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        h_prob, _ = self.sample_h(v)
        return h_prob

    def contrastive_divergence(self, v, lr=0.01):
        h_prob, h = self.sample_h(v)
        v_prob, _ = self.sample_v(h)
        h_prob2, _ = self.sample_h(v_prob)

        # Update weights and biases
        self.W.data += lr * (torch.mm(h.t(), v) - torch.mm(h_prob2.t(), v_prob)) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v_prob, dim=0)
        self.h_bias.data += lr * torch.mean(h - h_prob2, dim=0)

class DBN(nn.Module):
    def __init__(self, layers):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList([RBM(layers[i], layers[i+1]) for i in range(len(layers)-1)])

    def pretrain(self, data, epochs=10, lr=0.01):
        """
        Pre-trains the DBN layer by layer using Contrastive Divergence.
        """
        current_input = data
        for i, rbm in enumerate(self.rbm_layers):
            print(f"  Training RBM Layer {i+1}...")
            for epoch in range(epochs):
                rbm.contrastive_divergence(current_input, lr)
            current_input = rbm(current_input).detach() # Pass activations to next layer
        print("DBN pre-training complete.")

    def forward(self, x):
        """
        Performs a forward pass through the DBN to extract features.
        """
        for rbm in self.rbm_layers:
            x = rbm(x)
        return x

# ------------------------------ VAE ------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Sigmoid for output if input is normalized to [0,1]
        )

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE (encoder, reparameterization, decoder).
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

# ------------------------------ Training and Anomaly Detection Functions ------------------------------
def train_vae_dbn(train_data_path, epochs_dbn=10, epochs_vae=50, lr_dbn=0.01, lr_vae=0.001):
    """
    Trains the DBN for feature extraction and then the VAE for anomaly detection.
    Saves the trained VAE model.
    """
    print("\n--- Training VAE with DBN ---")
    X_train, _, _, encoders, scaler = load_and_preprocess_nslkdd_data(train_data_path)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    input_dim = X_train.shape[1]
    dbn_layers = [input_dim, 64, 32] # Example DBN layers
    dbn = DBN(dbn_layers)
    dbn.pretrain(X_train_tensor, epochs=epochs_dbn, lr=lr_dbn)

    # Extract features using trained DBN
    dbn_features = dbn(X_train_tensor).detach()
    print(f"DBN extracted features shape: {dbn_features.shape}")

    # Initialize and train VAE
    vae_model = VAE(dbn_features.shape[1]) # VAE input dim is the output dim of DBN
    optimizer = optim.Adam(vae_model.parameters(), lr=lr_vae)

    vae_model.train()
    print(f"Training VAE for {epochs_vae} epochs...")
    for epoch in range(epochs_vae):
        optimizer.zero_grad()
        recon, mu, logvar = vae_model(dbn_features)
        loss = vae_loss_function(recon, dbn_features, mu, logvar)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  VAE Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(vae_model.state_dict(), MODEL_PATH)
    print(f"VAE model saved to {MODEL_PATH}")

    # Calculate threshold for training data
    vae_model.eval()
    with torch.no_grad():
        recon, _, _ = vae_model(dbn_features)
        errors = torch.mean((dbn_features - recon) ** 2, dim=1).cpu().numpy()
        # Set threshold based on a percentile of training errors (e.g., 95th percentile)
        threshold = np.percentile(errors, 95)
    print(f"Anomaly threshold determined from training data: {threshold:.4f}")
    return vae_model, encoders, scaler, dbn, threshold

def detect_anomalies(model, dbn_model, data_path, encoders, scaler, threshold):
    """
    Loads test data, extracts features using DBN, and detects anomalies using VAE.
    Returns a list of detected anomaly reports.
    """
    print("\n--- Detecting Anomalies ---")
    X_test, y_test, original_df, _, _ = load_and_preprocess_nslkdd_data(data_path, encoders, scaler)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    dbn_features = dbn_model(X_test_tensor).detach()
    print(f"DBN extracted features for test data shape: {dbn_features.shape}")

    model.eval()
    with torch.no_grad():
        recon, _, _ = model(dbn_features)
        errors = torch.mean((dbn_features - recon) ** 2, dim=1).cpu().numpy()

    anomalies = errors > threshold
    results = []
    # Collect detailed anomaly reports
    for i in range(len(anomalies)):
        if anomalies[i]:
            row = original_df.iloc[i]
            # Construct a detailed anomaly report (you can customize fields)
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "src_ip": "192.168.1.100",  # Placeholder
                "dst_ip": "10.0.0.5",      # Placeholder
                "anomaly_type": "Network Traffic Anomaly",
                "severity": "High" if errors[i] > threshold * 1.5 else "Medium",
                "deviation_score": float(errors[i] * 100),
                "description": f"Detected abnormal network pattern with reconstruction error {errors[i]:.4f}",
                "action_taken": "Logged for review",
                "protocol": row.get('protocol_type', 'unknown') # Safely get protocol
            }
            formatted = "\n".join([f"  {key}: {value}" for key, value in result.items()])
            results.append(formatted + "\n")
    print(f"Detected {len(results)} anomalies.")
    return results, predicted, y_test.to_numpy()


def evaluate_detection(true_labels, predicted_labels):
    """
    Evaluates the anomaly detection performance using common metrics.
    """
    print("\n--- Evaluation Metrics ---")
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, zero_division=0)
    rec = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    return acc, prec, rec, f1

if __name__ == '__main__':
    # This block is for testing the VAE_DBN components in isolation
    print("Running VAE_DBN_model.py in standalone test mode.")

    # Ensure dummy data is available for testing or modify to use actual paths
    # For a quick test, we'll try to use the NSL-KDD paths if they exist
    if not os.path.exists(NSL_KDD_TRAIN_PATH):
        print(f"Warning: Training data not found at {NSL_KDD_TRAIN_PATH}. Cannot run standalone test.")
        print("Please ensure NSL-KDD datasets are placed in the 'data' directory or update paths.")
    else:
        # Train the model
        print("\n--- Standalone Training Test ---")
        trained_vae_model, encoders, scaler, dbn_trained, threshold = train_vae_dbn(NSL_KDD_TRAIN_PATH, epochs_dbn=2, epochs_vae=5) # Reduced epochs for testing

        # Load the model and perform detection
        print("\n--- Standalone Detection Test ---")
        if os.path.exists(MODEL_PATH):
            X_test, y_test, original_df, _, _ = load_and_preprocess_nslkdd_data(NSL_KDD_TEST_PATH, encoders, scaler)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            loaded_vae_model = VAE(X_test_tensor.shape[1] if dbn_trained is None else dbn_trained(X_test_tensor).shape[1])
            loaded_vae_model.load_state_dict(torch.load(MODEL_PATH))
            loaded_vae_model.eval()

            # For standalone test, we need to re-initialize DBN if not loaded from saved state
            # In actual integration, the DBN object will be passed from training
            # For this standalone test, we re-create a DBN with the same layer structure
            dbn_test_instance = DBN([X_test.shape[1], 64, 32])
            # Note: For a robust standalone test, DBN weights should also be saved/loaded
            # For simplicity here, we assume it's feature extraction which is stateless after pretraining
            # or pretrain it again on test data (not ideal, but works for isolated testing of VAE/DBN flow)
            dbn_test_instance.pretrain(X_test_tensor, epochs=1, lr=0.01) # Minimal pre-training for test

            results, predicted, true_labels = detect_anomalies(loaded_vae_model, dbn_test_instance, NSL_KDD_TEST_PATH, encoders, scaler, threshold)
            for r in results[:3]:
                print(r)
            evaluate_detection(true_labels, predicted)
        else:
            print("Model not saved, cannot perform standalone detection test.")