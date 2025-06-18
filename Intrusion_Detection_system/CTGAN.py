import pandas as pd
import torch
from sdv.single_table import CTGAN
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class CTGANSynthesizer:
    def __init__(self, epochs=300):
        self.epochs = epochs
        self.ctgan = None

    def train(self, data: pd.DataFrame):
        """
        Trains the CTGAN model on the provided real data.
        """
        print(f"Training CTGAN for {self.epochs} epochs...")
        self.ctgan = CTGAN(epochs=self.epochs, verbose=False) # verbose=False to reduce output
        self.ctgan.fit(data)
        print("CTGAN training complete.")

    def generate_synthetic_data(self, num_samples: int) -> pd.DataFrame:
        """
        Generates a specified number of synthetic samples using the trained CTGAN.
        """
        if self.ctgan is None:
            raise ValueError("CTGAN model not trained. Call train() first.")
        print(f"Generating {num_samples} synthetic samples...")
        synthetic_data = self.ctgan.sample(num_samples)
        print("Synthetic data generation complete.")
        return synthetic_data

    def evaluate_synthetic_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Placeholder for synthetic data evaluation.
        You can integrate advanced evaluation metrics here (e.g., DCR, visualization).
        """
        print("\n--- Evaluating Synthetic Data (Placeholder) ---")
        # Example: Basic comparison of statistical properties
        print("Real data shape:", real_data.shape)
        print("Synthetic data shape:", synthetic_data.shape)
        print("\nStatistical comparison (mean of first few columns):")
        print("Real Data Mean:\n", real_data.iloc[:, :5].mean())
        print("Synthetic Data Mean:\n", synthetic_data.iloc[:, :5].mean())
        print("Further evaluation metrics (e.g., DCR, visualizations) can be added here.")

if __name__ == '__main__':
    # This block is for testing the CTGANSynthesizer in isolation
    from src.utils import load_and_preprocess_ctgan_data

    print("Running CTGAN_model.py in standalone test mode.")
    # Create a dummy CSV for testing
    dummy_data = {
        'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'col2': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'col3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('dummy_data_ctgan.csv', index=False)

    # Load and preprocess dummy data
    train_df, _, _ = load_and_preprocess_ctgan_data('dummy_data_ctgan.csv')

    # Initialize and train CTGAN
    ctgan_synth = CTGANSynthesizer(epochs=10) # Reduced epochs for quick test
    ctgan_synth.train(train_df)

    # Generate synthetic data
    synthetic_df = ctgan_synth.generate_synthetic_data(num_samples=5)
    print("\nGenerated Synthetic Data Head:\n", synthetic_df.head())

    # Evaluate synthetic data
    ctgan_synth.evaluate_synthetic_data(train_df, synthetic_df)

    # Clean up dummy file
    import os
    os.remove('dummy_data_ctgan.csv')
    print("\nCTGAN standalone test complete.")