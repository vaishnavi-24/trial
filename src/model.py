"""
DistilBERT Model for Indirect Prompt Injection Detection
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertConfig,
    AutoConfig
)
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndirectPIDetector(nn.Module):
    """
    DistilBERT-based binary classifier for indirect prompt injection detection
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            model_name: Pre-trained model name
            num_labels: Number of classes (2 for binary classification)
            dropout: Dropout probability
            hidden_dim: Optional additional hidden layer dimension
        """
        super(IndirectPIDetector, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained DistilBERT
        logger.info(f"🔧 Loading {model_name}...")
        
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
        # Optional: Add custom classification head
        if hidden_dim is not None:
            self.custom_head = True
            self.pre_classifier = nn.Linear(self.config.hidden_size, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, num_labels)
            
            # Initialize weights
            self.pre_classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.pre_classifier.bias is not None:
                self.pre_classifier.bias.data.zero_()
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()
        else:
            self.custom_head = False
        
        logger.info(f"✅ Model loaded: {self.get_num_parameters():,} parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Dictionary with loss, logits, and predictions
        """
        if self.custom_head:
            # Use DistilBERT base model (without classification head)
            outputs = self.distilbert.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get [CLS] token representation
            hidden_state = outputs[0]  # [batch_size, seq_len, hidden_size]
            pooled_output = hidden_state[:, 0]  # [batch_size, hidden_size]
            
            # Custom classification head
            pooled_output = self.pre_classifier(pooled_output)
            pooled_output = nn.ReLU()(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        else:
            # Use standard DistilBERT for sequence classification
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': predictions
        }
    
    def get_num_parameters(self) -> int:
        """
        Count total trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_base_model(self):
        """
        Freeze DistilBERT base model (only train classification head)
        """
        logger.info("🔒 Freezing base DistilBERT model...")
        
        if self.custom_head:
            for param in self.distilbert.distilbert.parameters():
                param.requires_grad = False
        else:
            for param in self.distilbert.distilbert.parameters():
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✅ Trainable parameters: {trainable:,}")
    
    def unfreeze_last_n_layers(self, n: int = 2):
        """
        Unfreeze last n transformer layers for fine-tuning
        
        Args:
            n: Number of last layers to unfreeze
        """
        logger.info(f"🔓 Unfreezing last {n} transformer layers...")
        
        # DistilBERT has 6 transformer layers
        total_layers = len(self.distilbert.distilbert.transformer.layer)
        
        for i in range(total_layers - n, total_layers):
            for param in self.distilbert.distilbert.transformer.layer[i].parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✅ Trainable parameters: {trainable:,}")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Initialize model
    model = IndirectPIDetector(
        model_name="distilbert-base-uncased",
        num_labels=2,
        dropout=0.1
    )
    
    print(f"\n📊 Model Architecture:")
    print(model)
    
    print(f"\n🔢 Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    
    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        labels=dummy_labels
    )
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Predictions: {outputs['predictions']}")