# Bio-Contrast


# Drug Response Prediction Model

A deep learning framework for predicting drug responses in cancer cell lines and patient-derived organoids (PDOs) using gene expression data and transfer learning.

## Overview

This project implements a transfer learning approach to predict drug responses across different biological models. The framework consists of:

- Contrastive learning between cell lines and PDO data
- Hierarchical neural networks utilizing KEGG pathway information
- Drug response prediction models for both cell lines and PDOs
- Evaluation metrics for both regression (cell lines) and classification (PDOs) tasks


## Installation

Please use ```environment.yml``` to setup Anaconda environment.

## Project Structure

- `run.py` - Main script for training and evaluating models
- `data.py` - Data loading and preprocessing utilities
- `model.py` - Neural network model architectures 
- `modules.py` - Custom neural network modules and layers
- `trainer.py` - Training loop and evaluation functions
- `configuration.py` - Configuration management
- `utils.py` - Helper functions and loss calculations
- `hierarchies.py` - KEGG pathway hierarchy processing

## Configuration

The model configurations are specified in `config.yaml`. Key parameters include:

- Model architectures (encoders, predictors)
- Training parameters (learning rate, batch size)
- Network hyperparameters (hidden layers, dropout)

## Usage

To train a model:

```python run.py```

The script will:
1. Load cell line and PDO data
2. Train models for each drug
3. Perform cross-validation
4. Save results and model checkpoints

## Model Architecture

The framework consists of several key components:

### 1. Contrastive Learning Module
- Learns shared representations between cell lines and PDOs
- Uses temperature-scaled cross entropy loss
- Implements domain adaptation through contrastive learning

### 2. KEGG Hierarchical CNN
- Incorporates biological pathway information
- Uses hierarchical convolutional layers
- Processes gene expression data through pathway-guided architecture

### 3. Response Predictors
- Cell line response regression
- PDO response classification
