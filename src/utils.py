import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def calculate_class_distribution(y):
    """Calculate and return class distribution statistics."""
    total = len(y)
    fraud_count = np.sum(y == 1)
    non_fraud_count = np.sum(y == 0)
    
    return {
        'total_observations': total,
        'fraud_count': fraud_count,
        'non_fraud_count': non_fraud_count,
        'fraud_percentage': (fraud_count / total) * 100,
        'non_fraud_percentage': (non_fraud_count / total) * 100
    }

def preprocess_data(df, target_col, drop_cols=None):
    """
    Preprocess the data by:
    - Handling missing values
    - Dropping specified columns
    - Scaling numeric features
    - Separating features and target
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Drop specified columns
    if drop_cols:
        data = data.drop(columns=drop_cols)
    
    # Handle missing values
    # For simplicity, fill numeric columns with median and categorical with mode
    numeric_cols = data.select_dtypes(include=np.number).columns
    categorical_cols = data.select_dtypes(exclude=np.number).columns
    
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
            
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y

def plot_roc_curves(results_dict):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        plt.plot(
            results['roc']['fpr'], 
            results['roc']['tpr'], 
            label=f"{model_name} (AUC = {results['auc']:.3f})"
        )
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_confusion_matrices(results_dict, normalize=False):
    """Plot confusion matrices for multiple models."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd', 
            cmap='Blues',
            ax=ax
        )
        
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Non-Fraud', 'Fraud'])
        ax.set_yticklabels(['Non-Fraud', 'Fraud'])
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_names, importances, top_n=10):
    """Plot top N feature importances."""
    # Create DataFrame for easier sorting
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Take top N features
    top_features = feat_imp.head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=top_features
    )
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    
    return plt, feat_imp
