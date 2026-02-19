import os
import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, save_dir="checkpoints", logger=print):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        self.save_dir = save_dir
        self.logger = logger

    def step(self, fold,epoch, val_loss, val_acc, model, optimizer, scheduler,
             train_losses, val_losses, train_accuracies, val_accuracies):
        improved = False

        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            improved = True
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            improved = True

        if improved:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"model_fold_{fold + 1}.pth")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_accuracy': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
            torch.save(checkpoint, save_path)
            print(f"Model improved, saved to {save_path}")
            self.logger(f"✅ Fold {fold + 1} Epoch {epoch + 1}: Model improved, saved to {save_path}")
            self.no_improve_epochs = 0
            return False
        else:
            self.no_improve_epochs += 1
            # 早停
            if self.no_improve_epochs > self.patience:
                self.no_improve_epochs = 0
                return True
            print(f"No improvement. No_Improve_Epochs = {self.no_improve_epochs}/{self.patience}")
            self.logger(f"⏳ No improvement. Epochs without improvement: {self.no_improve_epochs}/{self.patience}")
            return False

