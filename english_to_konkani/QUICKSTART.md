# Quick Start Guide - English to Konkani Translation

**Complete beginner's guide** to get started with the English-to-Konkani translation system in 5 minutes!

---

## Step 1: Setup (2 minutes)

### Option A: Automated Setup (Recommended)

```bash
cd english_to_konkani
python setup.py
```

This will:
- ✅ Check Python version
- ✅ Install all dependencies
- ✅ Verify imports
- ✅ Check GPU availability
- ✅ Create directories

### Option B: Manual Setup

```bash
cd english_to_konkani
pip install -r requirements.txt
```

---

## Step 2: Check Dataset (30 seconds)

```bash
python app.py stats
```

**Output:**
```
==================================================
DATASET STATISTICS
==================================================
num_pairs: 25
eng_vocab_size: 82
kon_vocab_size: 45
...
==================================================
```

**Sample pairs already included!** You can train immediately.

---

## Step 3: Train Model (5-10 minutes)

### Quick Training (Small Dataset)

```bash
python app.py train --epochs 50
```

**Expected:**
- Training time: ~5 minutes (CPU) / ~1 minute (GPU)
- Final accuracy: 70-90% (with sample data)

### Production Training (Better Results)

```bash
python app.py train --epochs 200 --batch-size 32
```

**Expected:**
- Training time: ~15 minutes (CPU) / ~3 minutes (GPU)
- Final accuracy: 85-95%

---

## Step 4: Test Translation (1 minute)

### Single Sentence

```bash
python app.py translate --text "hello"
```

**Output:**
```
English: hello
Konkani: नमस्कार
```

### Multiple Sentences

Create test file:
```bash
echo "hello" > test.txt
echo "good morning" >> test.txt
echo "thank you" >> test.txt
```

Translate:
```bash
python app.py translate --input-file test.txt --output-file results.txt
```

View results:
```bash
type results.txt  # Windows
# cat results.txt  # Linux/Mac
```

---

## Step 5: Add More Data (Optional)

### Add Single Pair

```bash
python app.py add-pair --english "how are you" --konkani "तुम कसे आहात"
```

### Retrain (Incremental Learning)

```bash
python app.py train --epochs 20
```

The model will **continue learning** from where it left off!

---

## Integration with ISL Model

### Full Pipeline

**Terminal 1:** Run ISL gesture recognition
```bash
cd ../isl_to_english
python app.py run
```

**Terminal 2:** Auto-translate to Konkani
```bash
cd ../english_to_konkani
python app.py translate --monitor
```

Now:
1. Make hand signs in front of webcam
2. ISL model recognizes → writes to `isl_to_english.txt`
3. Translation model reads → translates → writes to `english_to_konkani.txt`

---

## Common Commands Cheat Sheet

```bash
# View help
python app.py --help

# View dataset stats
python app.py stats

# Train new model
python app.py train --new-model

# Resume training
python app.py train --epochs 50

# Translate text
python app.py translate --text "your text here"

# Translate from file
python app.py translate --input-file input.txt

# Monitor file continuously
python app.py translate --monitor

# Add translation pair
python app.py add-pair --english "eng" --konkani "kon"
```

---

## File Format Quick Reference

### konkani_pairs.txt
```
eng:hello
kon:नमस्कार

eng:good morning
kon:सुप्रभात
```

### isl_to_english.txt (input)
```
hello
good morning
thank you
```

### english_to_konkani.txt (output)
```
hello | नमस्कार
good morning | सुप्रभात
thank you | धन्यवाद
```

---

## Troubleshooting

### "Model not found" error
**Solution:** Train the model first
```bash
python app.py train
```

### Low accuracy
**Solution:** Add more training data
```bash
# Add 10-20 more pairs to konkani_pairs.txt
# Then retrain
python app.py train --epochs 100
```

### Slow training
**Solution:** Use GPU or reduce model size
```bash
python app.py train --embed-dim 64 --num-heads 2
```

---

## Need Help?

1. Check full README.md for detailed documentation
2. Run `python app.py --help` for command options
3. Open GitHub issue for bugs/questions

---

**You're all set!** Start translating English to Konkani 🎉
