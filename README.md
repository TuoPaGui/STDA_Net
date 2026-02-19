## STDA-Net

### STDA-Net: A Spectral–Temporal Dynamic Aware Network for Sleep Stage Classification with Multi-Channel EEG


## Abstract

Automatic sleep stage classification is essential for objective sleep quality assessment. This project implements **STDA-Net**, a spectral–temporal hybrid deep learning framework for multi-class sleep staging (W, N1, N2, N3, REM). The model integrates an STFT-based spectrogram front-end with physiologically-guided band enhancement and cross-band attention, followed by a dynamic multi-scale CNN backbone and an attention-guided temporal encoder combining multi-head self-attention with residual TCN modeling. In addition, the training pipeline supports dynamic batch-wise class reweighting, K-fold cross validation, early stopping, learning-rate scheduling, checkpoint resume, and detailed metric logging (Accuracy, Macro-F1, Cohen’s Kappa, confusion matrix).

## Requirements

* Python >= 3.8
* PyTorch >= 1.10
* NumPy
* scikit-learn
* tqdm
* librosa
* torchaudio
* kymatio
* torchdiffeq
* torch_geometric

## Prepare datasets

The model expects **preprocessed `.h5` EEG files**. Update the dataset path in the config file:

```
config/sda_net_config.json
```

Default path:

```
"./data/MASS/SS3/data_anno_savefiles"
```

## Training STDA-Net

All training and model hyperparameters are controlled by:

```
config/sda_net_config.json
```

You can modify:

* Batch size, epochs, patience
* K-fold number
* Optimizer and scheduler settings
* Loss settings (e.g., label smoothing)
* Model arguments (channels, dropout, etc.)

Run training with K-fold cross validation:

```
python trainer.py
```

## Results

The training pipeline produces:

* Saved best checkpoints per fold
* Training logs (loss, accuracy, F1, kappa)
* Confusion matrices (count + percent)
* Training/validation curves


