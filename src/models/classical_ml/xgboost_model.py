"""
XGBoost Model for Liver Fibrosis Staging
=========================================

This module implements XGBoost classifier for multi-class fibrosis staging (F0-F4)
using combined spatial and frequency domain features.

Model Architecture:
-------------------
Input: Spatial features + FFT features + NASH features
Output: Fibrosis stage (F0, F1, F2, F3, F4) or binary (F0-F2 vs F3-F4)

Author: Bülent Tuğrul
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                            confusion_matrix, classification_report, roc_curve)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import optuna


class FibrosisXGBoostModel:
    """
    XGBoost Classifier for Liver Fibrosis Staging
    
    Features:
    - Multi-class classification (F0-F4)
    - Binary classification (significant/advanced fibrosis)
    - Hyperparameter optimization with Optuna
    - SHAP explainability
    - Cross-validation
    """
    
    def __init__(self, 
                 task: str = 'multiclass',  # 'multiclass', 'binary_significant', 'binary_advanced'
                 use_gpu: bool = False,
                 random_state: int = 42):
        """
        Initialize XGBoost Model
        
        Parameters:
        -----------
        task : str
            Classification task:
            - 'multiclass': F0 vs F1 vs F2 vs F3 vs F4
            - 'binary_significant': F0-F1 vs F2-F4 (significant fibrosis)
            - 'binary_advanced': F0-F2 vs F3-F4 (advanced fibrosis)
        use_gpu : bool
            Use GPU acceleration if available
        random_state : int
            Random seed
        """
        self.task = task
        self.use_gpu = use_gpu
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.best_params = None
        
        # Default hyperparameters
        self.default_params = {
            'objective': 'multi:softprob' if task == 'multiclass' else 'binary:logistic',
            'num_class': 5 if task == 'multiclass' else None,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': random_state,
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'eval_metric': 'mlogloss' if task == 'multiclass' else 'logloss'
        }
        
        if task != 'multiclass':
            del self.default_params['num_class']
    
    def prepare_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Prepare labels based on task
        
        Parameters:
        -----------
        labels : np.ndarray
            Original fibrosis stages (F0-F4: 0-4)
            
        Returns:
        --------
        prepared_labels : np.ndarray
            Task-specific labels
        """
        if self.task == 'multiclass':
            return labels
        elif self.task == 'binary_significant':
            # F0-F1 (0-1) -> 0, F2-F4 (2-4) -> 1
            return (labels >= 2).astype(int)
        elif self.task == 'binary_advanced':
            # F0-F2 (0-2) -> 0, F3-F4 (3-4) -> 1
            return (labels >= 3).astype(int)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             feature_names: Optional[List[str]] = None,
             optimize_hyperparams: bool = False,
             n_trials: int = 50) -> Dict:
        """
        Train XGBoost model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels (F0-F4)
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
        feature_names : list, optional
            Feature names for interpretability
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        history : dict
            Training history
        """
        # Store feature names
        self.feature_names = feature_names
        
        # Prepare labels
        y_train_prepared = self.prepare_labels(y_train)
        if y_val is not None:
            y_val_prepared = self.prepare_labels(y_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            self.best_params = self._optimize_hyperparameters(
                X_train_scaled, y_train_prepared, n_trials
            )
            params = self.best_params
        else:
            params = self.default_params
        
        # Create eval set
        eval_set = [(X_train_scaled, y_train_prepared)]
        if X_val is not None:
            eval_set.append((X_val_scaled, y_val_prepared))
        
        # Train model
        print(f"Training XGBoost for {self.task} task...")
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            X_train_scaled, y_train_prepared,
            eval_set=eval_set,
            verbose=True
        )
        
        # Get training history
        history = {
            'train_loss': self.model.evals_result()['validation_0']['logloss'] if 'validation_0' in self.model.evals_result() else [],
        }
        
        if X_val is not None:
            history['val_loss'] = self.model.evals_result()['validation_1']['logloss'] if 'validation_1' in self.model.evals_result() else []
        
        print("Training completed!")
        return history
    
    def _optimize_hyperparameters(self, X, y, n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
            
            # Add task-specific params
            if self.task == 'multiclass':
                params['objective'] = 'multi:softprob'
                params['num_class'] = 5
                params['eval_metric'] = 'mlogloss'
            else:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            
            params['tree_method'] = 'gpu_hist' if self.use_gpu else 'hist'
            params['random_state'] = self.random_state
            
            # Cross-validation
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted' if self.task == 'multiclass' else 'f1')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest hyperparameters: {study.best_params}")
        print(f"Best CV score: {study.best_value:.4f}")
        
        # Combine with base params
        best_params = {**self.default_params, **study.best_params}
        return best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fibrosis stage
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        probabilities : np.ndarray
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        y_test_prepared = self.prepare_labels(y_test)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test_prepared, y_pred),
            'f1_score': f1_score(y_test_prepared, y_pred, average='weighted'),
        }
        
        # AUC for binary tasks
        if self.task != 'multiclass':
            metrics['auc'] = roc_auc_score(y_test_prepared, y_proba[:, 1])
        else:
            # Multi-class AUC (one-vs-rest)
            try:
                metrics['auc'] = roc_auc_score(y_test_prepared, y_proba, multi_class='ovr')
            except:
                metrics['auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test_prepared, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_test_prepared, y_pred, output_dict=True
        )
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        importance_df : pd.DataFrame
            Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def explain_with_shap(self, X_test: np.ndarray, 
                         max_display: int = 20,
                         save_path: Optional[str] = None):
        """
        Generate SHAP explanations
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test samples for explanation
        max_display : int
            Max features to display
        save_path : str, optional
            Path to save plot
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print("Computing SHAP values...")
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_scaled, 
                         feature_names=self.feature_names,
                         max_display=max_display,
                         show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP plot saved to {save_path}")
        
        plt.show()
        
        return shap_values
    
    def save_model(self, filepath: str):
        """Save model to file"""
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'task': self.task,
            'best_params': self.best_params
        }
        joblib.dump(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        save_dict = joblib.load(filepath)
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.task = save_dict['task']
        self.best_params = save_dict.get('best_params')
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("XGBoost Model for Liver Fibrosis Staging")
    print("="*80)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 100  # Combined FFT + Spatial + NASH features
    
    # Simulate features
    X = np.random.randn(n_samples, n_features)
    
    # Simulate fibrosis stages (F0-F4)
    y = np.random.randint(0, 5, n_samples)
    
    # Feature names
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model for multiclass
    print("\n" + "="*80)
    print("Training Multiclass Model (F0 vs F1 vs F2 vs F3 vs F4)")
    print("="*80)
    
    model_multi = FibrosisXGBoostModel(task='multiclass', use_gpu=False)
    history = model_multi.train(X_train, y_train, X_val, y_val, 
                                feature_names=feature_names)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model_multi.evaluate(X_test, y_test)
    
    print("\n" + "="*80)
    print("TEST SET RESULTS:")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_score']:.4f}")
    print(f"AUC (multiclass): {metrics['auc']:.4f}" if metrics['auc'] else "AUC: N/A")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP 10 IMPORTANT FEATURES:")
    print("="*80)
    importance_df = model_multi.get_feature_importance(top_n=10)
    print(importance_df.to_string(index=False))
    
    # Train binary model for significant fibrosis
    print("\n" + "="*80)
    print("Training Binary Model (Significant Fibrosis: F0-F1 vs F2-F4)")
    print("="*80)
    
    model_binary = FibrosisXGBoostModel(task='binary_significant')
    model_binary.train(X_train, y_train, X_val, y_val)
    
    metrics_binary = model_binary.evaluate(X_test, y_test)
    print(f"\nAccuracy: {metrics_binary['accuracy']:.4f}")
    print(f"F1-Score: {metrics_binary['f1_score']:.4f}")
    print(f"AUC: {metrics_binary['auc']:.4f}")
    
    print("\n" + "="*80)
    print("XGBoost Model Ready!")
    print("="*80)
