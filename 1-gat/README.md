# TAMER with WandB Integration

This repository includes support for tracking training runs with [Weights & Biases (WandB)](https://wandb.ai/). This allows you to monitor metrics like loss and accuracy in real-time, view validation predictions, and track system resources.

## Setup

### Environment Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```
This will install `wandb` and `pandas` along with other dependencies.

### WandB Authentication

There are 3 ways to authenticate, checked in this priority order:

1. **Kaggle Secrets (Recommended for Kaggle)**:
   - Go to your Kaggle Notebook -> Add-ons -> Secrets.
   - Add a new secret with Label: `wandb_api_key` and Value: `YOUR_WANDB_API_KEY`.
   - The script will automatically detect and use this.

2. **Environment Variable**:
   - Set `WANDB_API_KEY` in your environment.
   - Linux/Mac: `export WANDB_API_KEY=YOUR_KEY`
   - Windows (PowerShell): `$env:WANDB_API_KEY="YOUR_KEY"`

3. **Fallback (Hardcoded)**:
   - For convenience, a key has been temporarily hardcoded in `train.py`.
   - **SECURITY WARNING**: Please remove this key before making your code public or sharing it with others.

## Usage

Run the training script as usual. WandB will initialize and start logging automatically.

```bash
python train.py fit --config config.yaml
```

If you are just testing the integration, run a fast dev run:

```bash
python train.py fit --trainer.fast_dev_run=True
```

## Features

- **Metric Logging**: Automatically tracks `train_loss`, `val_loss`, `ExpRate`, etc.
- **System Monitoring**: CPU, GPU, Memory usage.
- **Rich Media**: In the WandB dashboard, verify the "Tables" or "Images" section. It will show a sample of 8 validation images per epoch with their **Ground Truth** and **Prediction** text, allowing you to visually verify model improvement.

## Notes for Kaggle

When running on Kaggle with "Save Version" > "Save & Run All", ensure internet access is enabled in the notebook settings so WandB can push logs.
