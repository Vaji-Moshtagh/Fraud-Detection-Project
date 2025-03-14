import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_class_distribution(y):
    """
    Plot the distribution of classes
    
    Parameters:
    -----------
    y : array-like
        Class labels
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Class (0=Normal, 1=Fraud)')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.show()
    
def scatter_2d_data(X, y, title="Transaction Data Distribution"):
    """
    Create a 2D scatter plot of the data after dimensionality reduction
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features (should be reduced to 2D)
    y : array-like
        Class labels
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different color
    for label in np.unique(y):
        plt.scatter(
            X[y == label, 0], 
            X[y == label, 1],
            label=f"Class #{label}",
            alpha=0.5,
            linewidth=0.15
        )
        
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
def compare_distributions(X_original, y_original, X_resampled, y_resampled, method_name="SMOTE"):
    """
    Compare original and resampled data distributions side by side
    
    Parameters:
    -----------
    X_original : numpy.ndarray
        Original features (2D)
    y_original : array-like
        Original labels
    X_resampled : numpy.ndarray
        Resampled features (2D)
    y_resampled : array-like
        Resampled labels
    method_name : str
        Name of resampling method used
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Original data plot
    for label in np.unique(y_original):
        ax1.scatter(
            X_original[y_original == label, 0],
            X_original[y_original == label, 1],
            label=f"Class #{label}",
            alpha=0.5,
            linewidth=0.15
        )
    ax1.set_title("Original set")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Resampled data plot
    for label in np.unique(y_resampled):
        ax2.scatter(
            X_resampled[y_resampled == label, 0],
            X_resampled[y_resampled == label, 1],
            label=f"Class #{label}",
            alpha=0.5,
            linewidth=0.15
        )
    ax2.set_title(method_name)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print class counts
    print("Original class distribution:")
    print(pd.Series(y_original).value_counts())
    print("\nResampled class distribution:")
    print(pd.Series(y_resampled).value_counts())
