import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE

class FraudDetector:
    """
    A class for detecting fraudulent transactions using various ML models
    and handling class imbalance.
    """
    
    def __init__(self, random_state=42):
        """Initialize the fraud detector with default configuration."""
        self.random_state = random_state
        self.models = {}
        self.test_size = 0.3
        
    def prepare_data(self, X, y):
        """Split data into training and test sets."""
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Ensures balanced class distribution in splits
        )
    
    def train_baseline_model(self, X_train, y_train):
        """Train a baseline Random Forest model without any special handling."""
        model = RandomForestClassifier(random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['baseline'] = model
        return model
    
    def train_smote_model(self, X_train, y_train):
        """Train a model with SMOTE sampling to handle class imbalance."""
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', RandomForestClassifier(random_state=self.random_state))
        ])
        pipeline.fit(X_train, y_train)
        self.models['smote'] = pipeline
        return pipeline
    
    def train_weighted_model(self, X_train, y_train, weight_ratio=10):
        """
        Train a model with class weighting to handle imbalance.
        Gives higher weight to the minority class.
        """
        model = RandomForestClassifier(
            class_weight={0: 1, 1: weight_ratio},
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['weighted'] = model
        return model
    
    def train_logistic_model(self, X_train, y_train):
        """Train a logistic regression model as an alternative approach."""
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        self.models['logistic'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance using multiple metrics."""
        # Get predictions
        y_pred = model.predict(X_test)
        
        # For models in a pipeline, need to use the right predict_proba method
        if isinstance(model, Pipeline):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate precision and recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Get statistics about natural accuracy
        total_observations = len(y_test)
        non_fraud_count = np.sum(y_test == 0)
        natural_accuracy = non_fraud_count / total_observations * 100
        
        return {
            'confusion_matrix': cm,
            'classification_report': cr,
            'roc': {'fpr': fpr, 'tpr': tpr},
            'auc': auc,
            'precision_recall': {'precision': precision, 'recall': recall},
            'natural_accuracy': natural_accuracy,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def get_feature_importance(self, model_name='baseline'):
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        # For pipelines, get the classifier
        if isinstance(self.models[model_name], Pipeline):
            model = self.models[model_name].named_steps['classifier']
        else:
            model = self.models[model_name]
            
        # Only works for models with feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")
