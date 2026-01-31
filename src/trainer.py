"""
Training Loop for Indirect Prompt Injection Detection
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import wandb
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndirectPITrainer:
    """
    Trainer for Indirect Prompt Injection Detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './models/checkpoints',
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        use_wandb: bool = False,
        wandb_project: str = 'indirect-pi-detection',
        wandb_run_name: Optional[str] = None
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            num_epochs: Number of training epochs
            warmup_ratio: Ratio of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            output_dir: Directory to save checkpoints
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate total steps
        self.total_steps = len(train_loader) * num_epochs
        self.warmup_steps = int(self.total_steps * warmup_ratio)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.best_eval_accuracy = 0.0
        self.training_history = []
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            run_name = wandb_run_name or f"distilbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    'model': model.model_name,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'num_epochs': num_epochs,
                    'batch_size': train_loader.batch_size,
                    'warmup_ratio': warmup_ratio,
                    'total_steps': self.total_steps,
                }
            )
            wandb.watch(model, log='all', log_freq=logging_steps)
        
        logger.info(f"🚀 Trainer initialized:")
        logger.info(f"   Device: {device}")
        logger.info(f"   Total steps: {self.total_steps:,}")
        logger.info(f"   Warmup steps: {self.warmup_steps:,}")
        logger.info(f"   Epochs: {num_epochs}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            predictions = outputs['predictions']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_predictions / total_predictions,
                'lr': f'{current_lr:.2e}'
            })
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_predictions
                
                log_dict = {
                    'train/loss': avg_loss,
                    'train/accuracy': accuracy,
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                    'train/step': self.global_step
                }
                
                if self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)
            
            # Evaluation
            if self.global_step % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Back to training mode
                
                # Save best model
                if eval_metrics['accuracy'] > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_metrics['accuracy']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(is_best=False)
        
        # Epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct_predictions / total_predictions
        }
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on evaluation set
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='binary',
            pos_label=1  # Malicious class
        )
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        eval_metrics = {
            'loss': total_loss / len(self.eval_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        # Log to console
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION @ Step {self.global_step}")
        logger.info(f"{'='*60}")
        logger.info(f"Loss: {eval_metrics['loss']:.4f}")
        logger.info(f"Accuracy: {eval_metrics['accuracy']*100:.2f}%")
        logger.info(f"Precision: {eval_metrics['precision']*100:.2f}%")
        logger.info(f"Recall: {eval_metrics['recall']*100:.2f}%")
        logger.info(f"F1-Score: {eval_metrics['f1']*100:.2f}%")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"{'='*60}\n")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'eval/loss': eval_metrics['loss'],
                'eval/accuracy': eval_metrics['accuracy'],
                'eval/precision': eval_metrics['precision'],
                'eval/recall': eval_metrics['recall'],
                'eval/f1': eval_metrics['f1']
            }, step=self.global_step)
        
        return eval_metrics
    
    def train(self) -> Dict[str, List]:
        """
        Full training loop
        
        Returns:
            Training history
        """
        logger.info(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(1, self.num_epochs + 1):
            # Train epoch
            epoch_metrics = self.train_epoch(epoch)
            
            logger.info(f"\nEpoch {epoch}/{self.num_epochs} Summary:")
            logger.info(f"  Train Loss: {epoch_metrics['loss']:.4f}")
            logger.info(f"  Train Accuracy: {epoch_metrics['accuracy']*100:.2f}%")
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Save history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': epoch_metrics['loss'],
                'train_accuracy': epoch_metrics['accuracy'],
                'eval_loss': eval_metrics['loss'],
                'eval_accuracy': eval_metrics['accuracy'],
                'eval_precision': eval_metrics['precision'],
                'eval_recall': eval_metrics['recall'],
                'eval_f1': eval_metrics['f1']
            })
            
            # Save end-of-epoch checkpoint
            self.save_checkpoint(is_best=False, epoch=epoch)
        
        logger.info("\n🎉 Training complete!")
        logger.info(f"Best evaluation accuracy: {self.best_eval_accuracy*100:.2f}%")
        
        # Save final training history
        self.save_training_history()
        
        if self.use_wandb:
            wandb.finish()
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
            epoch: Current epoch (optional)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_eval_accuracy': self.best_eval_accuracy,
            'training_history': self.training_history
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        # Save checkpoint
        if is_best:
            path = self.output_dir / 'best_model.pt'
            logger.info(f"💾 Saving best model to {path}")
        else:
            path = self.output_dir / f'checkpoint_step_{self.global_step}.pt'
        
        torch.save(checkpoint, path)
        
        # Also save model config and tokenizer
        if is_best:
            self.model.config.save_pretrained(self.output_dir / 'best_model_config')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"📂 Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_eval_accuracy = checkpoint['best_eval_accuracy']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"✅ Checkpoint loaded (step {self.global_step})")
    
    def save_training_history(self):
        """
        Save training history to JSON
        """
        history_path = self.output_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"💾 Training history saved to {history_path}")