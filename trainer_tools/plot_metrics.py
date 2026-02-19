import os
import matplotlib.pyplot as plt

SAVE_DIR = r"G:\Research\EEG_Project\Template\CodeDir\tools\training_validation_curves"

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=SAVE_DIR):
    # Plot train/val loss and accuracy, then save as PNG
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(train_losses, label="Train Loss")
    ax[0].plot(val_losses, label="Val Loss")
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(train_accuracies, label="Train Accuracy")
    ax[1].plot(val_accuracies, label="Val Accuracy")
    ax[1].set_title("Accuracy Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close(fig)
