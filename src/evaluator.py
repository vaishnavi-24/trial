"""
Comprehensive Evaluation for Indirect Prompt Injection Detection
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndirectPIEvaluator:
    """
    Comprehensive evaluation for indirect PI detection
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './results'
    ):
        """
        Args:
            model: Trained model
            tokenizer: Tokenizer
            device: Device to run evaluation on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
    
    def evaluate_dataset(
        self,
        test_loader: torch.utils.data.DataLoader,
        dataset_name: str = "Test"
    ) -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            test_loader: DataLoader for test set
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary with comprehensive metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING ON {dataset_name.upper()} DATASET")
        logger.info(f"{'='*80}")
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(outputs['logits'].cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Get probabilities
        probabilities = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
        positive_probs = probabilities[:, 1]  # Probability of malicious class
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels,
            all_predictions,
            positive_probs
        )
        
        # Print results
        self._print_results(metrics, dataset_name)
        
        # Save results
        self._save_results(metrics, dataset_name)
        
        # Plot visualizations
        self._plot_visualizations(
            all_labels,
            all_predictions,
            positive_probs,
            dataset_name
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """Calculate comprehensive metrics"""
        
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            matthews_corrcoef
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, pos_label=1),
            'recall': recall_score(labels, predictions, pos_label=1),
            'f1': f1_score(labels, predictions, pos_label=1),
            'mcc': matthews_corrcoef(labels, predictions),
            'roc_auc': roc_auc_score(labels, probabilities),
            'pr_auc': average_precision_score(labels, probabilities),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }
        
        # Per-class metrics
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _print_results(self, metrics: Dict, dataset_name: str):
        """Print formatted results"""
        
        print(f"\n📊 {dataset_name} Set Results:")
        print("="*80)
        print(f"Accuracy:     {metrics['accuracy']*100:.2f}%")
        print(f"Precision:    {metrics['precision']*100:.2f}%")
        print(f"Recall:       {metrics['recall']*100:.2f}%")
        print(f"F1-Score:     {metrics['f1']*100:.2f}%")
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
        print(f"MCC:          {metrics['mcc']:.4f}")
        print(f"\nSpecificity:  {metrics['specificity']*100:.2f}%")
        print(f"FPR:          {metrics['fpr']*100:.2f}%")
        print(f"FNR:          {metrics['fnr']*100:.2f}%")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              Benign  Malicious")
        print(f"Actual Benign    {metrics['true_negatives']:5d}  {metrics['false_positives']:5d}")
        print(f"      Malicious  {metrics['false_negatives']:5d}  {metrics['true_positives']:5d}")
        print("="*80)
    
    def _save_results(self, metrics: Dict, dataset_name: str):
        """Save results to file"""
        
        results_path = self.output_dir / f'{dataset_name.lower()}_results.json'
        
        # Convert numpy types to Python types
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                metrics_serializable[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        import json
        with open(results_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_path}")
    
    def _plot_visualizations(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        dataset_name: str
    ):
        """Create and save visualization plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[0, 0],
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious']
        )
        axes[0, 0].set_title(f'Confusion Matrix - {dataset_name}')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = roc_auc_score(labels, probabilities)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve - {dataset_name}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = average_precision_score(labels, probabilities)
        axes[1, 0].plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})', linewidth=2)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'Precision-Recall Curve - {dataset_name}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        benign_probs = probabilities[labels == 0]
        malicious_probs = probabilities[labels == 1]
        axes[1, 1].hist(benign_probs, bins=50, alpha=0.5, label='Benign', color='blue')
        axes[1, 1].hist(malicious_probs, bins=50, alpha=0.5, label='Malicious', color='red')
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('Predicted Probability (Malicious Class)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Prediction Distribution - {dataset_name}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{dataset_name.lower()}_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Plots saved to {plot_path}")
    
    def evaluate_per_scenario(
        self,
        test_dataset,
        scenario_column: str = 'scenario'
    ) -> pd.DataFrame:
        """
        Evaluate performance per scenario (Email, Web QA, etc.)
        
        Args:
            test_dataset: Test dataset with scenario labels
            scenario_column: Column name for scenario
            
        Returns:
            DataFrame with per-scenario metrics
        """
        logger.info("\n📊 Per-Scenario Evaluation:")
        
        results = []
        
        for scenario in test_dataset[scenario_column].unique():
            scenario_data = test_dataset.filter(
                lambda x: x[scenario_column] == scenario
            )
            
            # Create temporary dataloader
            from torch.utils.data import DataLoader
            scenario_loader = DataLoader(scenario_data, batch_size=32, shuffle=False)
            
            # Evaluate
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in scenario_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    all_predictions.extend(outputs['predictions'].cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results.append({
                'scenario': scenario,
                'count': len(all_labels),
                'accuracy': accuracy_score(all_labels, all_predictions),
                'precision': precision_score(all_labels, all_predictions, zero_division=0),
                'recall': recall_score(all_labels, all_predictions, zero_division=0),
                'f1': f1_score(all_labels, all_predictions, zero_division=0)
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("PER-SCENARIO PERFORMANCE")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Save
        results_df.to_csv(self.output_dir / 'per_scenario_results.csv', index=False)
        
        return results_df


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("✅ Evaluator module ready!")