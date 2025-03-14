import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_credit_data(filepath):
    """
    Load and preprocess credit card transaction data
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing credit card data
        
    Returns:
    --------
    df : pandas.DataFrame
        Processed dataframe
    """
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_features_labels(df, target_column='Class'):
    """
    Separate features and labels from the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
        
    Returns:
    --------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    """
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"Class distribution:\n{class_counts}")
    print(f"Class percentages:\n{class_counts / len(y) * 100}")
    
    return X, y

def scale_features(X):
    """
    Standardize features
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features
        
    Returns:
    --------
    X_scaled : numpy.ndarray
        Scaled features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
