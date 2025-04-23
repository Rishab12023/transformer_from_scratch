# Transformer from Scratch

This repository contains a simple and complete implementation of the Transformer architecture from scratch using PyTorch. It is intended for educational purposes to help understand how each component of the Transformer works.

## Contents

- `train.py` — Main script to train the Transformer model.
- `model_transformer.py` — Contains the model components, utility functions, and layers.
- `dataset.py` - Contains the dataset class for loading dataset from huggingface
- `config.py` - To set the different configuration associated with the model. Ex: Batch_Size, Dimensions, Vocab Size etc
- `notes/` — A collection of markdown files explaining core concepts used in the Transformer architecture, such as:
  - Positional Encoding
  - Multi-head Self Attention
  - Feedforward Networks
  - Learning Rate Scheduling
  - Label Smoothing

## How to Use

1. Install the required dependencies.
2. Run the training using:
   ```bash
   python train.py
   ```
3. Read the notes inside the `notes/` folder for a deeper understanding of each concept.

Feel free to explore and modify the code to gain a better understanding of the Transformer architecture.

