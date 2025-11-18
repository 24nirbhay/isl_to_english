"""
Command-line interface for English-Konkani Translation.
Beginner-friendly commands for training and translation.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import KonkaniPairsLoader
from train import TranslationTrainer
from translate import FileBasedTranslator


def cmd_stats(args):
    """Show dataset statistics."""
    loader = KonkaniPairsLoader(args.pairs_file)
    
    try:
        loader.load_pairs()
        stats = loader.get_statistics()
        
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 50 + "\n")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")


def cmd_train(args):
    """Train the translation model."""
    trainer = TranslationTrainer(
        pairs_file=args.pairs_file,
        model_dir=args.model_dir,
        max_eng_length=args.max_eng_length,
        max_kon_length=args.max_kon_length
    )
    
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        validation_split=args.val_split,
        resume_from_master=not args.new_model
    )


def cmd_translate(args):
    """Translate English to Konkani."""
    translator = FileBasedTranslator(
        model_path=os.path.join(args.model_dir, 'master', 'model.keras'),
        tokenizer_eng_path=os.path.join(args.model_dir, 'master', 'tokenizer_eng.pkl'),
        tokenizer_kon_path=os.path.join(args.model_dir, 'master', 'tokenizer_kon.pkl'),
        input_file=args.input_file,
        output_file=args.output_file,
        max_eng_length=args.max_eng_length,
        max_kon_length=args.max_kon_length
    )
    
    if args.text:
        # Translate command-line text
        translation = translator.translate(args.text)
        print(f"\nEnglish: {args.text}")
        print(f"Konkani: {translation}\n")
    
    elif args.monitor:
        # Monitor input file
        translator.monitor_file(check_interval=args.interval)
    
    else:
        # Translate entire file
        translator.translate_file()


def cmd_add_pair(args):
    """Add a new translation pair."""
    loader = KonkaniPairsLoader(args.pairs_file)
    loader.add_pair(args.english, args.konkani)
    print("\nPair added successfully!")
    print(f"Total pairs: {len(loader.eng_sentences) + 1}")


def main():
    parser = argparse.ArgumentParser(
        description="English-to-Konkani Neural Machine Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset statistics
  python app.py stats

  # Train the model
  python app.py train --epochs 100

  # Translate a sentence
  python app.py translate --text "hello"

  # Translate from file
  python app.py translate --input-file isl_to_english.txt

  # Monitor file for new sentences
  python app.py translate --monitor

  # Add new translation pair
  python app.py add-pair --english "hello" --konkani "नमस्कार"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--pairs-file', default='konkani_pairs.txt', help='Translation pairs file')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train translation model')
    train_parser.add_argument('--pairs-file', default='konkani_pairs.txt', help='Translation pairs file')
    train_parser.add_argument('--model-dir', default='models', help='Model directory')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension')
    train_parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    train_parser.add_argument('--ff-dim', type=int, default=512, help='Feed-forward dimension')
    train_parser.add_argument('--num-encoder-layers', type=int, default=2, help='Number of encoder layers')
    train_parser.add_argument('--num-decoder-layers', type=int, default=2, help='Number of decoder layers')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    train_parser.add_argument('--val-split', type=float, default=0.15, help='Validation split')
    train_parser.add_argument('--max-eng-length', type=int, default=100, help='Max English sequence length')
    train_parser.add_argument('--max-kon-length', type=int, default=100, help='Max Konkani sequence length')
    train_parser.add_argument('--new-model', action='store_true', help='Train new model (don\'t resume from master)')
    train_parser.set_defaults(func=cmd_train)
    
    # Translate command
    translate_parser = subparsers.add_parser('translate', help='Translate English to Konkani')
    translate_parser.add_argument('--model-dir', default='models', help='Model directory')
    translate_parser.add_argument('--text', type=str, help='Text to translate')
    translate_parser.add_argument('--input-file', default='isl_to_english.txt', help='Input file')
    translate_parser.add_argument('--output-file', default='english_to_konkani.txt', help='Output file')
    translate_parser.add_argument('--monitor', action='store_true', help='Monitor input file continuously')
    translate_parser.add_argument('--interval', type=float, default=1.0, help='Check interval (seconds)')
    translate_parser.add_argument('--max-eng-length', type=int, default=100, help='Max English sequence length')
    translate_parser.add_argument('--max-kon-length', type=int, default=100, help='Max Konkani sequence length')
    translate_parser.set_defaults(func=cmd_translate)
    
    # Add pair command
    add_parser = subparsers.add_parser('add-pair', help='Add new translation pair')
    add_parser.add_argument('--pairs-file', default='konkani_pairs.txt', help='Translation pairs file')
    add_parser.add_argument('--english', required=True, help='English sentence')
    add_parser.add_argument('--konkani', required=True, help='Konkani translation')
    add_parser.set_defaults(func=cmd_add_pair)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
