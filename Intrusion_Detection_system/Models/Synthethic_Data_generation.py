import pandas as pd
import torch
from sdv.single_table import CTGANSynthesizer
    # sdc is synthetic data vault used to generate synthetic data. CTGAN is a type of GAN (Generative Adversarial Network) specifically designed for tabular data.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sdv.metadata import SingleTableMetadata 

class   MyCTGANSynthesizer:
    def __init__(self, epochs=300):   # the default epoch is set to 300. while the calling of this constructor if the user gives some other value so that value will be overriden by the default value of 300.
        self.epochs = epochs
        self.ctgan = None

    def train(self, data: pd.DataFrame):
        # Trains the CTGAN model on the provided real data. pd.DataFrame is the real data.

        print(f"Training CTGAN for {self.epochs} epochs...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        self.ctgan = CTGANSynthesizer(metadata=metadata, epochs=self.epochs)# verbose=False to reduce output. Instantiates the CTGAN model from the sdv library.
        self.ctgan.fit(data) # this is the core training step where the generator and discriminator plays the game of mouse and rat. you know what im talking about.
        print("CTGAN training complete.") # now the generator should generate synthetic data that would be so real.

    def generate_synthetic_data(self, num_samples: int) -> pd.DataFrame:
        
        # Generates a specified number of synthetic samples using the trained CTGAN. the num_samples is the number of samples to generate.
        # -> pd.DataFrame: Indicates that the method will return a pandas DataFrame
        
        if self.ctgan is None:
            raise ValueError("CTGAN model not trained. Call train() first.")
        print(f"Generating {num_samples} synthetic samples...")
        synthetic_data = self.ctgan.sample(num_samples) # this is where the magic happens. the trained CTGAN model generates synthetic data based on the learned distribution of the real data.
        print("Synthetic data generation complete.")
        return synthetic_data
    
    # Till here the synthethic data generation is done. Now we can evaluate the synthetic data. for the future evalutation  
    
    # the evaluation part starts below.
    # def evaluate_synthetic_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    #     """
    #     Placeholder for synthetic data evaluation.
    #     You can integrate advanced evaluation metrics here (e.g., DCR, visualization).
    #     """
    #     print("\n--- Evaluating Synthetic Data (Placeholder) ---")
    #     # Example: Basic comparison of statistical properties
    #     print("Real data shape:", real_data.shape)
    #     print("Synthetic data shape:", synthetic_data.shape)
    #     print("\nStatistical comparison (mean of first few columns):")
    #     print("Real Data Mean:\n", real_data.iloc[:, :5].mean())
    #     print("Synthetic Data Mean:\n", synthetic_data.iloc[:, :5].mean())
    #     print("Further evaluation metrics (e.g., DCR, visualizations) can be added here.")
    # the evaluation part ends above. the evaluation part is commented out for now. you can uncomment it and use it later. 

# now this file is responsible for generating synthetic data using CTGAN. The CTGANSynthesizer class encapsulates the functionality for training a CTGAN model, generating synthetic data, and evaluating the synthetic data against real data. 
# this class is being used and called in the main.py file where the CTGAN model is trained on the real data and synthetic data is generated.