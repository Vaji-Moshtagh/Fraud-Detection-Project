from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def apply_pca(X, n_components=2):
    """
    Apply PCA for dimensionality reduction
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features
    n_components : int
        Number of components to keep
        
    Returns:
    --------
    X_pca : numpy.ndarray
        Reduced features
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return X_pca

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance the dataset
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features
    y : array-like
        Class labels
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_resampled : numpy.ndarray
        Resampled features
    y_resampled : array-like
        Resampled labels
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
