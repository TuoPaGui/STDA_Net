# Import model
# Import data generation tool
from data_processing import data_labels_generate
# Import training utilities
from trainer_tools import calculate_steps
from trainer_tools import trainer_log
from trainer_tools import plot_metrics
from trainer_tools import metrics
from trainer_tools import early_stopping
from trainer_tools import get_batch_class_weights
# Deep learning framework
import torch
import torch.optim as optim  # Optimizer
from torch.optim import lr_scheduler  # Scheduler
# ===== Data and system libraries =====
import glob  # File matching module for loading .h5 files
import random  # Python random utility (for shuffling data)
from sklearn.model_selection import KFold  # K-fold cross validation splitter
import os  # System operations (path, file I/O)
import numpy as np  # Numerical computation (labels, weights processing)
import torch.nn as nn  # Neural network modules (Linear, Loss, Conv, etc.)
import json  # Read config file (config.json)
import importlib
from tqdm import tqdm  # Progress bar


def log_message(message):
    log_file_path = os.path.join("G:\\log_path", 'model_training_log.txt')
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


# Load configuration file
with open("./config/sda_net_config.json", 'r') as f:
    config = json.load(f)


class Trainer:
    def __init__(self):

        # ========= Read configuration =========

        # Training parameters
        self.trainer_cfg = config["trainer"]
        self.seed = self.trainer_cfg["seed"]
        self.seed2 = self.trainer_cfg["seed2"]
        self.num_epochs = self.trainer_cfg["num_epochs"]
        self.k_folds = self.trainer_cfg["k_folds"]
        self.batches = self.trainer_cfg["batch_size"]
        self.patience = self.trainer_cfg["patience"]

        # Paths
        self.path_cfg = config["paths"]
        self.data_files_path = self.path_cfg["data_dir"]
        self.model_path = self.path_cfg["model_path"]
        self.log_dir = self.path_cfg["log_dir"]
        self.csv_output_dir = self.path_cfg["csv_output_dir"]

        # ========= Build model and components =========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config["model"]
        self.model_args = model_cfg["args"]
        module = importlib.import_module(f"model.{model_cfg['module']}")
        self.ModelClass = getattr(module, model_cfg["type"])
        self.model = self.ModelClass(**self.model_args).to(self.device)

        # Optimizer
        optimizer_type = config["optimizer"]["type"]
        self.opt_args = config["optimizer"]["args"]
        self.OptimizerClass = getattr(optim, optimizer_type)
        self.optimizer = self.OptimizerClass(self.model.parameters(), **self.opt_args)

        # Learning rate scheduler
        scheduler_type = config["scheduler"]["type"]
        self.sched_args = config["scheduler"]["args"]
        self.SchedulerClass = getattr(lr_scheduler, scheduler_type)
        self.scheduler = self.SchedulerClass(self.optimizer, **self.sched_args)

        # Loss function
        loss_type = config["loss"]["type"]
        self.loss_args = config["loss"]["args"]
        self.LossClass = getattr(nn, loss_type)
        self.criterion = self.LossClass(**self.loss_args)

        # Related variable definitions
        self.k_fold_splitter = None
        self.epoch = 0
        self.fold = 0
        self.train_files = []
        self.val_files = []
        self.train_steps = 0
        self.val_steps = 0
        self.best_train_metrics = {'accuracy': -float('inf'), 'f1': -float('inf'), 'kappa': -float('inf')}
        self.best_val_metrics = {'accuracy': -float('inf'), 'f1': -float('inf'), 'kappa': -float('inf')}
        self.num_classes = 5
        self.start_epoch = 0
        self.last_weights = None  # Used for dynamic class weight calculation

        # Data path
        self.data_files = glob.glob(os.path.join(self.data_files_path, "*.h5"))

        # Initialize data generator
        self.data_labels_generator = data_labels_generate.DataLabelsGenerate()
        self.train_data_generator = None
        self.val_data_generator = None

        # Used for monitoring training performance
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Early stopping variables
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.no_improve_epochs = 0
        self.early_stopper = early_stopping.EarlyStopping(
            patience=self.patience,
            save_dir=self.model_path,
            logger=log_message
        )

    def data_init(self):

        # Python and NumPy seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        # PyTorch CPU and GPU seeds
        torch.manual_seed(self.seed2)
        torch.cuda.manual_seed(self.seed2)
        torch.cuda.manual_seed_all(self.seed2)

        random.shuffle(self.data_files)
        self.k_fold_splitter = KFold(n_splits=self.k_folds, shuffle=False)
        # self.k_fold_splitter = KFold(n_splits=self.k_folds, shuffle=True, random_state=int(time.time()))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialization before each fold
    def fold_init(self, fold):
        self.fold = fold
        # Reset early stopping parameters
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.no_improve_epochs = 0
        self.best_train_metrics = {'accuracy': -float('inf'), 'f1': -float('inf'), 'kappa': -float('inf')}
        self.best_val_metrics = {'accuracy': -float('inf'), 'f1': -float('inf'), 'kappa': -float('inf')}

        # Reset model and optimizer before each fold
        self.model = self.ModelClass(**self.model_args).to(self.device)

        self.optimizer = self.OptimizerClass(self.model.parameters(), **self.opt_args)
        self.scheduler = self.SchedulerClass(self.optimizer, **self.sched_args)
        self.criterion = self.LossClass(**self.loss_args)

        self.early_stopper = early_stopping.EarlyStopping(patience=self.patience, save_dir=self.model_path, logger=log_message)
        print(f"Training fold {fold + 1}/{self.k_folds}...")
        log_message(f"Training fold {fold + 1}/{self.k_folds}...")

    def load_checkpoint(self, fold):
        model_path = os.path.join(self.model_path, f"model_fold_{self.fold + 1}.pth")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_val_acc = checkpoint.get('best_val_accuracy', 0.0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            print(f"✅ Loaded checkpoint from {model_path}, resuming from epoch {self.start_epoch}")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint for fold {fold + 1}: {e}")
            self.start_epoch = 0
            return False

    def train_per_epoch(self):
        epoch_loss = 0.0
        train_preds, train_labels = [], []
        self.model.train()
        progress_bar = tqdm(range(self.train_steps), desc=f"Epoch {self.epoch + 1}", ncols=100)

        for _ in progress_bar:
            batch = next(self.train_data_generator)
            data, labels = batch['batch_data'].to(self.device), batch['labels'].to(self.device)

            # Compute dynamic class weights
            batch_weights = get_batch_class_weights.get_batch_class_weights(
                labels, num_classes=self.num_classes, device=self.device, prev_weights=self.last_weights
            )
            self.last_weights = batch_weights.detach()
            loss_args = self.loss_args.copy()
            loss_args["weight"] = batch_weights
            # Update class weights (comment out to use default mean reduction)
            self.criterion = self.LossClass(**loss_args)

            self.optimizer.zero_grad()  # Clear previous gradients
            outputs = self.model(data, labels, self.criterion)

            loss = outputs['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
            self.optimizer.step()  # Update parameters (weights and biases)
            epoch_loss += loss.item()

            train_preds.extend(torch.argmax(outputs['logits'], 1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'Loss': loss.item()})

        return epoch_loss / self.train_steps, train_preds, train_labels

    def validate_per_epoch(self):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            progress_bar = tqdm(range(self.val_steps), desc=f"Validation Epoch {self.epoch + 1}", ncols=100)
            for _ in progress_bar:
                batch = next(self.val_data_generator)
                data, labels = batch['batch_data'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(data, labels, self.criterion)
                loss = outputs['total_loss']
                val_loss += loss.item()

                preds = torch.argmax(outputs['logits'], 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'Val Loss': loss.item()})

                # Identify poorly performing files
                if self.epoch > 16:
                    if loss > 0.7:
                        log_message(f"Bad file: {batch['file_name']} Loss: {loss}")

        print(f"Ground truth: {all_labels}")
        print(f"Predictions : {all_preds}")
        return val_loss / self.val_steps, all_preds, all_labels

    def model_train_epochs(self):
        # Data initialization
        self.data_init()

        # Check if there is a previous resume state
        resume_fold = 0
        if os.path.exists("resume_state.json"):
            with open("resume_state.json", "r") as f:
                resume_fold = int(f.read().strip())
            print(f"🔁 Resume from fold {resume_fold + 1}")
        else:
            print("🚀 Starting from fold 1")

        # Use all files for K-fold validation
        # train_val_files, test_files = train_test_split(self.data_files, test_size=0.1, random_state=42)
        train_val_files = self.data_files

        for fold, (train_index, val_index) in enumerate(self.k_fold_splitter.split(train_val_files)):
            if fold < resume_fold:
                continue  # Skip completed folds

            # Initialization before fold training
            self.fold_init(fold)

            self.train_files = [train_val_files[i] for i in train_index]
            self.val_files = [train_val_files[i] for i in val_index]

            self.train_steps = calculate_steps.calculate_steps_per_epoch(self.train_files, self.batches)
            self.val_steps = calculate_steps.calculate_steps_per_epoch(self.val_files, self.batches)

            log_message(f"train_files: {self.train_files}")
            log_message(f"val_files: {self.val_files}")

            self.train_data_generator = self.data_labels_generator.generate_data_labels(self.train_files, self.batches)
            self.val_data_generator = self.data_labels_generator.generate_data_labels(self.val_files, self.batches)

            # Load checkpoint
            self.load_checkpoint(fold)

            for self.epoch in range(self.start_epoch, self.num_epochs):
                # Training phase
                train_loss, train_preds, train_labels = self.train_per_epoch()
                train_metrics = metrics.calculate_metrics(train_preds, train_labels)

                # Validation phase
                val_loss, val_preds, val_labels = self.validate_per_epoch()
                val_metrics = metrics.calculate_metrics(val_preds, val_labels)

                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Print metrics
                print(f"\nFold {fold + 1} Epoch {self.epoch + 1}/{self.num_epochs}")
                print(f"Current Learning Rate: {current_lr:.6f}")
                print(f"[Train] Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | Kappa: {train_metrics['kappa']:.4f}")
                print(f"[ Val ] Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | Kappa: {val_metrics['kappa']:.4f}")

                # Save best metrics
                if train_metrics['accuracy'] > self.best_train_metrics['accuracy']:
                    self.best_train_metrics = train_metrics
                if val_metrics['accuracy'] > self.best_val_metrics['accuracy']:
                    self.best_val_metrics = val_metrics

                # Log per-epoch training info
                log_message(f"{'*' * 40} {self.epoch + 1} {'*' * 40}")
                log_message("Evaluation Results:")
                log_message(f"Val_Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | Kappa: {val_metrics['kappa']:.4f}")
                class_names = ["W", "N1", "N2", "N3", "REM"]
                trainer_log.log_confusion_matrix_with_metrics(
                    train_metrics['confusion_matrix_percent'], train_metrics['precision'], train_metrics['f1_score'],
                    train_metrics['recall'], class_names, title="Train"
                )
                trainer_log.log_confusion_matrix_with_metrics(
                    val_metrics['confusion_matrix_percent'], val_metrics['precision'], val_metrics['f1_score'],
                    val_metrics['recall'], class_names, title="Val"
                )
                log_message(f"Train_Loss: {train_loss:.4f} | ")
                log_message(f"Val_Loss: {val_loss:.4f} | ")
                log_message(f"Train_Accuracy: {train_metrics['accuracy']:.4f} | ")
                log_message(f"Val_Accuracy: {val_metrics['accuracy']:.4f} | ")
                log_message(f"Best_train_accuracy : {self.best_train_metrics['accuracy']:.4f} |")
                log_message(f"Best_val_accuracy : {self.best_val_metrics['accuracy']:.4f} |")

                # Update metric storage
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_metrics['accuracy'])
                self.val_accuracies.append(val_metrics['accuracy'])

                # Update curves in real time
                plot_metrics.plot_metrics(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

                # Early stopping check
                should_stop = self.early_stopper.step(
                    fold=fold,
                    epoch=self.epoch,
                    val_loss=val_loss,
                    val_acc=val_metrics['accuracy'],
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    train_losses=self.train_losses,
                    val_losses=self.val_losses,
                    train_accuracies=self.train_accuracies,
                    val_accuracies=self.val_accuracies
                )
                if should_stop:
                    with open("resume_state.json", "w") as file:
                        file.write(str(fold))
                    break

        log_message(f"[Best Train] Acc: {self.best_train_metrics['accuracy']:.4f} | F1: {self.best_train_metrics['f1']:.4f} | Kappa: {self.best_train_metrics['kappa']:.4f}")
        log_message(f"[ Best Val ] Acc: {self.best_val_metrics['accuracy']:.4f} | F1: {self.best_val_metrics['f1']:.4f} | Kappa: {self.best_val_metrics['kappa']:.4f}")
        log_message("\n Training completed!")
        # Remove fold resume state file
        if os.path.exists("resume_state.json"):
            os.remove("resume_state.json")
            print("🧹 Training complete, removed resume_state.json")
