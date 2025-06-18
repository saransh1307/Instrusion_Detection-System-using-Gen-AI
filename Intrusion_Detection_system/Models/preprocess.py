import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Intrusion_Detection_system.Models.ctgan import CTGAN
ctgan = CTGAN(epochs=100)

np.random.seed(42)
torch.manual_seed(42)

# Step 1: Load and preprocess data
def load_and_preprocess_data(filepath=None, df_input=None, encoders=None, scaler=None):
    if df_input is None:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(',') for line in lines if '?' not in line]
        df = pd.DataFrame(lines)
        df.columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        df = df.drop(columns=['difficulty'])
        original_df = df.copy()
    else:
        df = df_input.copy()
        original_df = df.copy()

    categorical_cols = ['protocol_type', 'service', 'flag', 'attack_type']
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col])

    X = df.drop(columns=['attack_type']).astype(np.float32)
    y = df['attack_type']

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, original_df, encoders, scaler

# Step 2: Deep Belief Network (DBN)
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        _, h_sample = self.v_to_h(v)
        return h_sample

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            p_h, h = self.v_to_h(v0)
            p_v, v0 = self.h_to_v(h)
        return v, v0.detach()

    def train_rbm(self, train_data, epochs=5, batch_size=64, lr=0.01):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for data, _ in train_loader:
                v, v_model = self.contrastive_divergence(data)
                loss = torch.mean((v - v_model) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

class DBN(nn.Module):
    def __init__(self, layer_sizes):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([
            RBM(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for rbm in self.rbms:
            x = rbm(x)
        return x

    def pretrain(self, data, epochs=5, batch_size=64, lr=0.01):
        input_data = data
        for rbm in self.rbms:
            rbm.train_rbm(TensorDataset(input_data, torch.zeros(input_data.size(0))), epochs, batch_size, lr)
            input_data = rbm(input_data).detach()

# Step 3: Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc21 = nn.Linear(128, latent_size)
        self.fc22 = nn.Linear(128, latent_size)
        self.fc3 = nn.Linear(latent_size, 128)
        self.fc4 = nn.Linear(128, input_size)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# Step 4: Prepare data for CTGAN

def prepare_data_for_ctgan(original_df):
    df_ctgan = original_df.copy()
    if 'difficulty' in df_ctgan.columns:
        df_ctgan = df_ctgan.drop(columns=['difficulty'])
    categorical_cols = ['protocol_type', 'service', 'flag', 'attack_type']
    for col in categorical_cols:
        df_ctgan[col] = df_ctgan[col].astype(str)
    return df_ctgan

# Step 5: Main pipeline

def main():
    filepath = 'KDDTrain+.txt'  # Replace with actual path
    X, y, original_df, encoders, scaler = load_and_preprocess_data(filepath)

    # Train CTGAN on attack-only data to generate synthetic attack samples
    df_ctgan = prepare_data_for_ctgan(original_df)
    ctgan = CTGAN(epochs=100)
    ctgan.fit(df_ctgan[df_ctgan['attack_type'] != 'normal'])
    synthetic_data = ctgan.sample(10000)

    # Merge and preprocess combined dataset
    combined_df = pd.concat([original_df, synthetic_data], ignore_index=True)
    X_combined, y_combined, _, _, _ = load_and_preprocess_data(df_input=combined_df, encoders=encoders, scaler=scaler)
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)

    # Pretrain DBN
    dbn = DBN([X_tensor.size(1), 128, 64])
    dbn.pretrain(X_tensor, epochs=5)
    features = dbn(X_tensor).detach()

    # Train VAE
    vae = VAE(input_size=64, latent_size=32)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    for epoch in range(10):
        recon_batch, mu, logvar = vae(features)
        loss = vae.loss_function(recon_batch, features, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()
