"""
MAIN TRAINING SCRIPT
Complete training pipeline for Indirect Prompt Injection Detection using DistilBERT
"""

import torch
import argparse
import logging
import os
from pathlib import Path

# Import custom modules
from src.data_loader import IndirectPIDataLoader, create_dataloaders
from src.model import IndirectPIDetector
from src.trainer import IndirectPITrainer
from src.evaluator import IndirectPIEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train DistilBERT for Indirect Prompt Injection Detection'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for data caching')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Proportion for test set (0.0-1.0)')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Proportion of remaining data for validation (0.0-1.0)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='Pre-trained model name')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='./models/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    
    # W&B logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='indirect-pi-detection',
                       help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation (skip training)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    
    # Parse arguments
    args = parse_args()
    
    logger.info("="*80)
    logger.info("INDIRECT PROMPT INJECTION DETECTION - TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("="*80)
    
    # ========================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ========================================
    
    logger.info("\n📥 STEP 1: Loading and preprocessing data...")
    
    data_loader = IndirectPIDataLoader(
        model_name=args.model_name,
        max_length=args.max_length,
        cache_dir=args.data_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42
    )
    
    # Load raw dataset
    raw_dataset = data_loader.load_dataset()
    
    # Get statistics
    stats = data_loader.get_data_statistics()
    logger.info("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Print sample examples
    data_loader.print_examples(num_examples=2)
    
    # Preprocess dataset
    processed_dataset = data_loader.process_dataset()
    
    # Create dataloaders (now returns 3: train, val, test)
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"\n✅ Data loading complete!")
    logger.info(f"   Training batches: {len(train_loader)}")
    logger.info(f"   Validation batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    # ========================================
    # STEP 2: INITIALIZE MODEL
    # ========================================
    
    logger.info("\n🧠 STEP 2: Initializing model...")
    
    model = IndirectPIDetector(
        model_name=args.model_name,
        num_labels=2,
        dropout=args.dropout
    )
    
    logger.info(f"✅ Model initialized: {model.get_num_parameters():,} parameters")
    
    # ========================================
    # STEP 3: TRAINING
    # ========================================
    
    if not args.eval_only:
        logger.info("\n🚀 STEP 3: Training model...")
        
        trainer = IndirectPITrainer(
            model=model,
            train_loader=train_loader,
            eval_loader=val_loader,  # Use validation set during training
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name
        )
        
        # Train!
        training_history = trainer.train()
        
        logger.info("\n🎉 Training complete!")
        logger.info(f"   Best accuracy: {trainer.best_eval_accuracy*100:.2f}%")
        
        # Load best model for evaluation
        best_model_path = Path(args.output_dir) / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded best model from {best_model_path}")
    
    else:
        logger.info("\n⏭️  Skipping training (eval_only mode)")
        
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded checkpoint from {args.checkpoint_path}")
    
    # ========================================
    # STEP 4: COMPREHENSIVE EVALUATION
    # ========================================
    
    logger.info("\n📊 STEP 4: Comprehensive evaluation...")
    
    evaluator = IndirectPIEvaluator(
        model=model,
        tokenizer=data_loader.tokenizer,
        device=args.device,
        output_dir='./results'
    )
    
    # Evaluate on test set (final holdout)
    test_metrics = evaluator.evaluate_dataset(
        test_loader=test_loader,
        dataset_name="BIPIA_Test"
    )
    
    # Per-scenario evaluation (if scenario column exists)
    if 'scenario' in raw_dataset['test'].column_names:
        logger.info("\n📊 Evaluating per scenario...")
        per_scenario_results = evaluator.evaluate_per_scenario(
            test_dataset=raw_dataset['test'],
            scenario_column='scenario'
        )
    
    logger.info("\n" + "="*80)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📁 Outputs saved to:")
    logger.info(f"   Models: {args.output_dir}")
    logger.info(f"   Results: ./results")
    logger.info(f"\n✅ Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"✅ Final Test F1-Score: {test_metrics['f1']*100:.2f}%")
    logger.info("\n✅ All done!")


if __name__ == "__main__":
    main()