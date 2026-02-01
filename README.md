# NLP NER Playground

An educational playground for exploring and experimenting with multilingual Named Entity Recognition (NER) using the XTREME benchmark and transformer models.

> ⚠️ **Note**: This is a learning and experimentation project, not intended for production use.

## 🐍 Python Version

This project uses **Python 3.11.10**

## Description

This educational project explores fine-tuning multilingual models for Named Entity Recognition (NER) using:

- The **XTREME** benchmark from Hugging Face (PAN-X subset)
- The **XLM-RoBERTa** model (xlm-roberta-base)
- A zero-shot cross-lingual transfer approach

### Main Use Case

The project focuses on a **Swiss multilingual** corpus including:
- 🇩🇪 German (62%)
- 🇫🇷 French (22%)
- 🇮🇹 Italian (8%)
- 🇬🇧 English (5%)

The goal is to train a model on German and evaluate its performance on other languages without additional training (zero-shot transfer).

## Features

- Loading and preprocessing XTREME dataset (PAN-X)
- Tokenization with SentencePiece (XLM-RoBERTa)
- Custom token classification model implementation
- Fine-tuning with 🤗 Transformers Trainer
- Evaluation with seqeval metrics (precision, recall, F1-score)
- Error analysis and loss visualization per token

## Installation

### Prerequisites

- Python 3.11.10
- CUDA (optional, for GPU acceleration)

### Installing Dependencies

```bash
pip install -r req.txt
```

> **Important**: If you're resuming work after a break or using different dependency versions, you may encounter errors when loading the custom model (especially in cell 18). The custom model architecture is tightly coupled to specific versions of `transformers` and `torch`. If you experience issues:
> 1. Ensure you're using the exact versions specified in `req.txt`
> 2. Consider retraining the model with your current environment versions
> 3. Alternatively, use the pre-trained model from Hugging Face Hub which handles version compatibility automatically

### Main Dependencies

- `transformers==4.56.2` - NLP models and pipelines
- `datasets==4.1.1` - Dataset loading and management
- `torch==2.7.1` - Deep learning framework
- `seqeval==1.2.2` - Metrics for sequence labeling
- `pandas==2.3.2` - Data manipulation
- `jupyter==1.1.1` - Interactive notebook

## Usage

### Launch the Notebook

```bash
jupyter notebook main.ipynb
```

### Main Steps

1. **Dataset Loading**: Import PAN-X corpus for target languages
2. **Data Exploration**: Analysis of entity distribution (LOC, PER, ORG)
3. **Tokenization**: Data preparation with XLM-RoBERTa tokenizer
4. **Fine-tuning**: Training on German corpus
5. **Evaluation**: Testing on other languages (French, Italian, English)
6. **Error Analysis**: Identification of problematic tokens

### Inference Example

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("RyuXiu/xlm-roberta-base-finetuned-panx-de")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

text = "Jeff Dean ist ein Informatiker bei Google in Kalifornien"
# ... use tag_text() to get predictions
```

## Data Format

The project uses the **IOB** (Inside-Outside-Beginning) format for labeling:

- `B-{TYPE}`: Beginning of an entity
- `I-{TYPE}`: Inside an entity
- `O`: Outside any entity

Supported entity types: `PER` (Person), `LOC` (Location), `ORG` (Organization)

## Model Architecture

The custom model inherits from `RobertaPreTrainedModel` and includes:

1. **Model Body**: XLM-RoBERTa base (12 layers, 768 dimensions)
2. **Classification Head**: 
   - Dropout layer
   - Linear layer (768 → num_labels)

## Metrics

Evaluation with `seqeval`:
- **Precision**: Proportion of correctly predicted entities
- **Recall**: Proportion of actual entities detected
- **F1-score**: Harmonic mean of precision and recall

## Training Configuration

```python
num_epochs = 3
batch_size = 24
weight_decay = 0.01
eval_strategy = "epoch"
```

## Contributing

This is an educational playground for learning and experimentation. Feel free to:
- Test other models (mT5, mDeBERTaV3)
- Experiment with other languages
- Improve error analysis
- Optimize hyperparameters
- Try different training strategies
- Explore various datasets

## Notes

- This is a **learning project** designed for educational purposes and experimentation
- The fine-tuned model is available on Hugging Face Hub: `RyuXiu/xlm-roberta-base-finetuned-panx-de`
- Training was performed on Google Colab to leverage GPU resources
- Labels `-100` are used to ignore special tokens and sub-words in loss calculation
- **Version Compatibility**: Custom model implementations may break with different versions of `transformers` or `torch`. Always use the versions specified in `req.txt` or retrain the model

## Resources

- [XTREME Benchmark](https://huggingface.co/datasets/xtreme)
- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [seqeval Documentation](https://github.com/chakki-works/seqeval)

## License

Educational and experimental project - for learning purposes only.

---

**Author**: Ryan  
**Python Version**: 3.11.10  
**Last Updated**: 2026
