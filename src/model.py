from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : array-like
        Labels
    test_size : float
        Proportion of test data
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Split datasets
    """
    return train_test_split(X, y, test_size=test_size, 
                            random_state=random_state, stratify=y)

def train_evaluate_model(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Train and evaluate a Random Forest model
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training data
    X_test, y_test : numpy.ndarray
        Test data
    use_smote : bool
        Whether to use SMOTE for resampling
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    from imblearn.over_sampling import SMOTE
    
    # Apply SMOTE if requested
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Applied SMOTE resampling")
        print(f"Training with {len(y_train)} samples")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return model
