import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Intrusion_Detection_system.utils import load_and_preprocess_nslkdd_data, vae_loss_function

# Constants specific to VAE/DBN model
MODEL_PATH = "Intrusion_Detection_system/models/vae_model_with_dbn.pth"
NSL_KDD_TRAIN_PATH = "Intrusion_Detection_system/DATA/RAW/KDDTrain+.txt"
NSL_KDD_TEST_PATH = "Intrusion_Detection_system/DATA/RAW/KDDTest+.txt"

# ------------------------------ DBN ------------------------------
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden): #it initializes the RBM with visible and hidden units. 
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):   #this function answers the question: Should each hidden neuron turn on given this input from visible layer?
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias)) # this computes the hidden layer probabilities from given visible layer activations.
        # h_prob is the probability of each hidden neuron being activated (firing) or not. 
        return h_prob, torch.bernoulli(h_prob) #this samples 0 or 1 from the Bernoulli distribution based on the probabilities. When we sample from torch.bernoulli(h_prob), we are tossing this coin to decide whether the hidden neuron fires (turns ON = 1) or not.

    def sample_v(self, h): # this function answers the question: Given this hidden pattern, what visible input does the model reconstruct?
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))  #v_prob is the probability of each visible neuron being activated (firing) or not.
        return v_prob, torch.bernoulli(v_prob) # this samples 0 or 1 from the Bernoulli distribution based on the probabilities. When we sample from torch.bernoulli(v_prob), we are tossing this coin to decide whether the visible neuron fires (turns ON = 1) or not.

    def forward(self, v):
        h_prob, _ = self.sample_h(v)
        return h_prob

    def contrastive_divergence(self, v, lr=0.01): # This function performs one step of contrastive divergence to update the weights and biases.
        h_prob, h = self.sample_h(v)
        v_prob, _ = self.sample_v(h)
        h_prob2, _ = self.sample_h(v_prob)

        self.W.data += lr * (torch.mm(h.t(), v) - torch.mm(h_prob2.t(), v_prob)) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v_prob, dim=0)
        self.h_bias.data += lr * torch.mean(h - h_prob2, dim=0)

class DBN(nn.Module):
    def __init__(self, layers): # DBM is stack of RBMs. layers is a list of integers where each integer represents the number of neurons in that layer.
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList([RBM(layers[i], layers[i+1]) for i in range(len(layers)-1)])

    def pretrain(self, data, epochs=10, lr=0.01): # This function pretrains the DBN using contrastive divergence on each RBM layer. Each layer is trained sequentially. Each RBM layer is trained on the output of the previous layer.
        current_input = data
        for i, rbm in enumerate(self.rbm_layers):
            print(f"  Training RBM Layer {i+1}...")
            for epoch in range(epochs):
                rbm.contrastive_divergence(current_input, lr)
            current_input = rbm(current_input).detach()
        print("DBN pre-training complete.")

    def forward(self, x): # This function passes the input through all RBM layers sequentially to get final DBN features.
        for rbm in self.rbm_layers:
            x = rbm(x)
        return x

# ------------------------------ VAE ------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(      # This is the encoder part of the VAE which compresses the input data into a latent space.
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
 
        self.decoder = nn.Sequential( # This is the decoder part of the VAE which reconstructs the input data from the latent space.
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar): # This function implements the reparameterization trick to sample from the latent space.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): # This function defines the forward pass of the VAE.
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

# ------------------------------ Training and Anomaly Detection Functions ------------------------------
def train_vae_dbn(NSL_KDD_TRAIN_PATH, epochs_dbn=10, epochs_vae=50, lr_dbn=0.01, lr_vae=0.001):
    print("\n--- Training VAE with DBN ---")
    X_train, _, _, encoders, scaler = load_and_preprocess_nslkdd_data(NSL_KDD_TRAIN_PATH)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    input_dim = X_train.shape[1]
    dbn_layers = [input_dim, 64, 32]
    dbn = DBN(dbn_layers)
    dbn.pretrain(X_train_tensor, epochs=epochs_dbn, lr=lr_dbn)

    dbn_features = dbn(X_train_tensor).detach()
    print(f"DBN extracted features shape: {dbn_features.shape}")

    vae_model = VAE(dbn_features.shape[1])
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

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(vae_model.state_dict(), MODEL_PATH)
    print(f"VAE model saved to {MODEL_PATH}")

    vae_model.eval()
    with torch.no_grad():
        recon, _, _ = vae_model(dbn_features)
        errors = torch.mean((dbn_features - recon) ** 2, dim=1).cpu().numpy()
        threshold = np.percentile(errors, 95)
    print(f"Anomaly threshold determined from training data: {threshold:.4f}")
    return vae_model, encoders, scaler, dbn, threshold

def detect_anomalies(model, dbn_model, data_path, encoders, scaler, threshold):
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
    for i in range(len(anomalies)):
        if anomalies[i]:
            row = original_df.iloc[i]
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "src_ip": "192.168.1.100",    # the NSLKDD dataset does not have src_ip or des_ip, so we use a placeholder.
                "dst_ip": "10.0.0.5",
                "anomaly_type": "Network Traffic Anomaly",
                "severity": "High" if errors[i] > threshold * 1.5 else "Medium",
                "deviation_score": float(errors[i] * 100),
                "description": f"Detected abnormal network pattern with reconstruction error {errors[i]:.4f}",
                "action_taken": "Logged for review",
                "protocol": row.get('protocol_type', 'unknown')
            }
            formatted = "\n".join([f"  {key}: {value}" for key, value in result.items()])
            results.append(formatted + "\n")
    print(f"Detected {len(results)} anomalies.")
    predicted = anomalies.astype(int)
    return results, predicted, y_test.to_numpy()

def evaluate_detection(true_labels, predicted_labels):
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



# below is the standalone test code to run the VAE and DBN model training and detection without needing to integrate with the main application.
# if __name__ == '__main__':
#     print("Running VAE_DBN_model.py in standalone test mode.")

#     if not os.path.exists(NSL_KDD_TRAIN_PATH):
#         print(f"Warning: Training data not found at {NSL_KDD_TRAIN_PATH}. Cannot run standalone test.")
#         print("Please ensure NSL-KDD datasets are placed in the 'data' directory or update paths.")
#     else:
#         print("\n--- Standalone Training Test ---")
#         trained_vae_model, encoders, scaler, dbn_trained, threshold = train_vae_dbn(NSL_KDD_TRAIN_PATH, epochs_dbn=2, epochs_vae=5)

#         print("\n--- Standalone Detection Test ---")
#         if os.path.exists(MODEL_PATH):
#             X_test, y_test, original_df, _, _ = load_and_preprocess_nslkdd_data(NSL_KDD_TEST_PATH, encoders, scaler)
#             X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

#             loaded_vae_model = VAE(dbn_trained(X_test_tensor).shape[1])
#             loaded_vae_model.load_state_dict(torch.load(MODEL_PATH))
#             loaded_vae_model.eval()

#             dbn_test_instance = DBN([X_test.shape[1], 64, 32])
#             dbn_test_instance.pretrain(X_test_tensor, epochs=1, lr=0.01)

#             results, predicted, true_labels = detect_anomalies(loaded_vae_model, dbn_test_instance, NSL_KDD_TEST_PATH, encoders, scaler, threshold)
#             for r in results[:3]:
#                 print(r)
#             evaluate_detection(true_labels, predicted)
#         else:
#             print("Model not saved, cannot perform standalone detection test.")
