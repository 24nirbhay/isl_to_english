"""
Training script for English-Konkani NMT model.
Supports incremental learning and beginner-friendly workflow.
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

from data_loader import KonkaniPairsLoader
from tokenizer import SimpleTokenizer, prepare_translation_data
from model_architecture import create_transformer_model


class TranslationTrainer:
    """
    Handles training of the translation model.
    Supports incremental learning with master model approach.
    """
    
    def __init__(
        self,
        pairs_file: str = "konkani_pairs.txt",
        model_dir: str = "models",
        max_eng_length: int = 100,
        max_kon_length: int = 100
    ):
        """
        Initialize trainer.
        
        Args:
            pairs_file: Path to translation pairs file
            model_dir: Directory to save models
            max_eng_length: Maximum English sequence length
            max_kon_length: Maximum Konkani sequence length
        """
        self.pairs_file = pairs_file
        self.model_dir = model_dir
        self.max_eng_length = max_eng_length
        self.max_kon_length = max_kon_length
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'master'), exist_ok=True)
        
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        validation_split: float = 0.1,
        resume_from_master: bool = True
    ):
        """
        Train the translation model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            validation_split: Validation split ratio
            resume_from_master: Whether to resume from master model
        """
        print("=" * 60)
        print("ENGLISH-TO-KONKANI TRANSLATION MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        print("\n1. Loading translation pairs...")
        loader = KonkaniPairsLoader(self.pairs_file)
        eng_sentences, kon_sentences = loader.load_pairs()
        
        # Show statistics
        stats = loader.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Prepare data
        print("\n2. Preparing data...")
        encoder_input, decoder_input, decoder_output, eng_tokenizer, kon_tokenizer = prepare_translation_data(
            eng_sentences, kon_sentences,
            max_eng_length=self.max_eng_length,
            max_kon_length=self.max_kon_length
        )
        
        # Create or load model
        print("\n3. Creating model...")
        
        master_model_path = os.path.join(self.model_dir, 'master', 'model.keras')
        master_tokenizer_eng_path = os.path.join(self.model_dir, 'master', 'tokenizer_eng.pkl')
        master_tokenizer_kon_path = os.path.join(self.model_dir, 'master', 'tokenizer_kon.pkl')
        
        if resume_from_master and os.path.exists(master_model_path):
            print("Loading existing master model...")
            model = keras.models.load_model(master_model_path, compile=False)
            print("Loaded master model from previous training")
        else:
            print("Creating new model...")
            model = create_transformer_model(
                eng_vocab_size=eng_tokenizer.vocab_size,
                kon_vocab_size=kon_tokenizer.vocab_size,
                max_eng_length=self.max_eng_length,
                max_kon_length=self.max_kon_length,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dropout=dropout
            )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nModel Summary:")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")
        
        # Create timestamped model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_model_dir = os.path.join(self.model_dir, f'model_{timestamp}')
        os.makedirs(current_model_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(current_model_dir, 'model.keras'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                filename=os.path.join(current_model_dir, 'training_history.csv')
            )
        ]
        
        # Train model
        print("\n4. Training model...")
        print(f"Training samples: {len(encoder_input)}")
        print(f"Validation split: {validation_split}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}\n")
        
        history = model.fit(
            [encoder_input, decoder_input],
            decoder_output,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        print("\n5. Saving model...")
        model.save(os.path.join(current_model_dir, 'model.keras'))
        print(f"Model saved to {current_model_dir}")
        
        # Save tokenizers
        eng_tokenizer.save(os.path.join(current_model_dir, 'tokenizer_eng.pkl'))
        kon_tokenizer.save(os.path.join(current_model_dir, 'tokenizer_kon.pkl'))
        print("Tokenizers saved")
        
        # Update master model
        print("\n6. Updating master model...")
        model.save(master_model_path)
        eng_tokenizer.save(master_tokenizer_eng_path)
        kon_tokenizer.save(master_tokenizer_kon_path)
        print("Master model updated")
        
        # Print training summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Model directory: {current_model_dir}")
        print(f"Master model: {master_model_path}")
        
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\nFinal Metrics:")
        print(f"  Training Loss: {final_loss:.4f}")
        print(f"  Training Accuracy: {final_acc:.4f}")
        print(f"  Validation Loss: {final_val_loss:.4f}")
        print(f"  Validation Accuracy: {final_val_acc:.4f}")
        
        return model, eng_tokenizer, kon_tokenizer


if __name__ == "__main__":
    # Simple training script
    trainer = TranslationTrainer(
        pairs_file="../konkani_pairs.txt",
        model_dir="../models"
    )
    
    trainer.train(
        epochs=100,
        batch_size=16,
        embed_dim=128,
        num_heads=4,
        ff_dim=512,
        num_encoder_layers=2,
        num_decoder_layers=2,
        validation_split=0.15
    )
