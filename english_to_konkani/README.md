# English to Konkani Neural Machine Translation

A **beginner-friendly** Transformer-based neural machine translation system that converts English sentences to Konkani (Devanagari script).

## Features

✅ **Simple File-Based I/O** - Read from `konkani_pairs.txt` with `eng:`/`kon:` prefix format  
✅ **Incremental Learning** - Add new translation pairs and retrain without losing previous knowledge  
✅ **File Monitoring** - Auto-translate sentences from `isl_to_english.txt` in real-time  
✅ **Lightweight Architecture** - Character-level tokenization, small model for CPU/GPU  
✅ **Beginner-Friendly** - Clear code structure, easy-to-use CLI, comprehensive documentation

## Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

The translation pairs are stored in `konkani_pairs.txt` with this format:

```
eng:hello
kon:नमस्कार

eng:good morning
kon:सुप्रभात

eng:thank you
kon:धन्यवाद
```

**Sample file already included!** You can start training immediately.

### 3. Train Model

```bash
# Train with default settings (100 epochs, batch size 16)
python app.py train

# Custom training
python app.py train --epochs 200 --batch-size 32 --val-split 0.2
```

**Training progress:**
- Model checkpoints saved to `models/model_YYYYMMDD_HHMMSS/`
- Best model automatically saved as `models/master/model.keras`
- Training history logged to CSV

### 4. Translate

```bash
# Translate a single sentence
python app.py translate --text "hello"

# Translate from file
echo "hello" > isl_to_english.txt
python app.py translate --input-file isl_to_english.txt

# Monitor file for continuous translation
python app.py translate --monitor
```

### 5. Add New Pairs

```bash
# Add a new translation pair
python app.py add-pair --english "how are you" --konkani "तुम कसे आहात"

# Retrain with new data (incremental learning)
python app.py train
```

## Project Structure

```
english_to_konkani/
├── app.py                      # CLI interface
├── konkani_pairs.txt           # Translation dataset (eng:/kon: format)
├── isl_to_english.txt          # Input file (from ISL model)
├── english_to_konkani.txt      # Output file (translations)
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── src/
│   ├── data_loader.py          # Load konkani_pairs.txt
│   ├── tokenizer.py            # Character-level tokenization
│   ├── model_architecture.py   # Transformer model
│   ├── train.py                # Training pipeline
│   └── translate.py            # Inference engine
│
└── models/
    ├── master/                 # Latest trained model
    │   ├── model.keras
    │   ├── tokenizer_eng.pkl
    │   └── tokenizer_kon.pkl
    └── model_YYYYMMDD_HHMMSS/  # Timestamped checkpoints
```

## Model Architecture

**Simplified Transformer** for beginner-friendliness:

- **Encoder:** 2 layers, 128-dim embeddings, 4 attention heads
- **Decoder:** 2 layers with masked self-attention + cross-attention
- **Tokenization:** Character-level (simple, no complex preprocessing)
- **Parameters:** ~500K (small enough for CPU training)

**Why this architecture?**
- Easy to understand and modify
- Fast training on limited data
- No complex dependencies (spaCy, sentencepiece, etc.)
- Works well for short sentences

## Usage Examples

### View Dataset Statistics

```bash
python app.py stats
```

Output:
```
==================================================
DATASET STATISTICS
==================================================
num_pairs: 25
eng_vocab_size: 82
kon_vocab_size: 45
avg_eng_length: 2.3
avg_kon_length: 1.8
==================================================
```

### Train New Model

```bash
python app.py train --epochs 50 --new-model
```

### Resume Training (Incremental)

```bash
# Add more pairs
python app.py add-pair --english "goodbye" --konkani "निरोप"

# Continue training from master model
python app.py train --epochs 20
```

### Batch Translation

```bash
# Create input file
echo "hello" > input.txt
echo "good morning" >> input.txt
echo "thank you" >> input.txt

# Translate all at once
python app.py translate --input-file input.txt --output-file output.txt

# View results
cat output.txt
```

## Integration with ISL Model

This translation module integrates seamlessly with the ISL-to-English system:

1. **ISL Model** recognizes hand signs → outputs English to `isl_to_english.txt`
2. **This Module** reads from `isl_to_english.txt` → translates to Konkani
3. **Output** written to `english_to_konkani.txt`

**Example workflow:**

```bash
# Terminal 1: Run ISL real-time translator
cd ../isl_to_english
python app.py run

# Terminal 2: Monitor and translate
cd ../english_to_konkani
python app.py translate --monitor
```

## File Format Specification

### Input: `konkani_pairs.txt`

```
eng:<english_sentence>
kon:<konkani_translation>

eng:<next_english_sentence>
kon:<next_konkani_translation>
```

**Rules:**
- Each pair must have `eng:` followed by `kon:` on the next line
- Blank lines are allowed (for readability)
- UTF-8 encoding required for Devanagari script

### Input: `isl_to_english.txt`

```
hello
good morning
thank you
```

Plain text file with one English sentence per line.

### Output: `english_to_konkani.txt`

```
hello | नमस्कार
good morning | सुप्रभात
thank you | धन्यवाद
```

Format: `<english> | <konkani>`

## Advanced Configuration

### Model Hyperparameters

```bash
python app.py train \
  --epochs 200 \
  --batch-size 32 \
  --embed-dim 256 \
  --num-heads 8 \
  --ff-dim 1024 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --dropout 0.2 \
  --val-split 0.2
```

### Custom Paths

```bash
python app.py train \
  --pairs-file my_custom_pairs.txt \
  --model-dir my_models/

python app.py translate \
  --model-dir my_models/ \
  --input-file my_input.txt \
  --output-file my_output.txt
```

## Future: Multimodal Training

**Planned features** (not yet implemented):

- **Audio:** Konkani speech → text transcription → training pairs
- **Video:** Subtitle extraction → training pairs
- **Text:** Already supported ✅

## Troubleshooting

### Model not found error

```
Error: Model not found: models/master/model.keras
```

**Solution:** Train the model first with `python app.py train`

### Unicode errors

```
UnicodeDecodeError: 'charmap' codec can't decode byte...
```

**Solution:** Ensure files are saved with UTF-8 encoding

### Low accuracy

- **Add more training data** to `konkani_pairs.txt`
- **Increase epochs:** `--epochs 200`
- **Increase model size:** `--embed-dim 256 --num-heads 8`

## Requirements

- Python 3.8+
- TensorFlow 2.17.0
- NumPy 1.26.4
- Scikit-learn 1.5.2

See `requirements.txt` for exact versions.

## License

MIT License - feel free to use for learning and research!

## Contributing

Contributions welcome! Areas for improvement:

- Add more Konkani translation pairs
- Implement audio/video preprocessing
- Optimize model architecture
- Add web interface

## Contact

For questions or issues, please open a GitHub issue.
