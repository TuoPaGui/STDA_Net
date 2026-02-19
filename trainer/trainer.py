from trainer_tools import data_labels_generate
from trainer_tools import calculate_steps
from trainer_tools import trainer_log
from trainer_tools import plot_metrics
from trainer_tools import metrics
from trainer_tools import early_stopping
from trainer_tools import get_batch_class_weights

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import glob
import random
from sklearn.model_selection import KFold
import os
import numpy as np
import torch.nn as nn
import json
import importlib
from tqdm import tqdm


def log_message(message):
    log_file_path = os.path.join("G:\\Log_Path", "log.txt")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


with open("./config/stda_net_config.json", "r") as f:
    config = json.load(f)


class Trainer:
    def __init__(self):

        # Training configuration
        self.trainer_cfg = config["trainer"]
        self.seed = self.trainer_cfg["seed"]
        self.seed2 = self.trainer_cfg["seed2"]
        self.num_epochs = self.trainer_cfg["num_epochs"]
        self.k_folds = self.trainer_cfg["k_folds"]
        self.batches = self.trainer_cfg["batch_size"]
        self.patience = self.trainer_cfg["patience"]

        # Path configuration
        self.path_cfg = config["paths"]
        self.data_files_path = self.path_cfg["data_dir"]
        self.model_path = self.path_cfg["model_path"]
        self.log_dir = self.path_cfg["log_dir"]
        self.csv_output_dir = self.path_cfg["csv_output_dir"]

        # Build model and components
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

        # Scheduler
        scheduler_type = config["scheduler"]["type"]
        self.sched_args = config["scheduler"]["args"]
        self.SchedulerClass = getattr(lr_scheduler, scheduler_type)
        self.scheduler = self.SchedulerClass(self.optimizer, **self.sched_args)

        # Loss function
        loss_type = config["loss"]["type"]
        self.loss_args = config["loss"]["args"]
        self.LossClass = getattr(nn, loss_type)
        self.criterion = self.LossClass(**self.loss_args)

        # Training state variables
        self.k_fold_splitter = None
        self.epoch = 0
        self.fold = 0
        self.train_files = []
        self.val_files = []
        self.train_steps = 0
        self.val_steps = 0
        self.best_train_metrics = {"accuracy": -float("inf"), "f1": -float("inf"), "kappa": -float("inf")}
        self.best_val_metrics = {"accuracy": -float("inf"), "f1": -float("inf"), "kappa": -float("inf")}
        self.num_classes = 5
        self.start_epoch = 0
        self.last_weights = None

        self.data_files = glob.glob(os.path.join(self.data_files_path, "*.h5"))

        self.data_labels_generator = data_labels_generate.DataLabelsGenerate()
        self.train_data_generator = None
        self.val_data_generator = None

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.no_improve_epochs = 0

        self.early_stopper = early_stopping.EarlyStopping(
            patience=self.patience,
            save_dir=self.model_path,
            logger=log_message,
        )

    def data_init(self):

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed2)
        torch.cuda.manual_seed(self.seed2)
        torch.cuda.manual_seed_all(self.seed2)

        random.shuffle(self.data_files)

        self.k_fold_splitter = KFold(n_splits=self.k_folds, shuffle=False)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fold_init(self, fold):
        self.fold = fold

        # Reset early stopping and metrics
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.no_improve_epochs = 0
        self.best_train_metrics = {"accuracy": -float("inf"), "f1": -float("inf"), "kappa": -float("inf")}
        self.best_val_metrics = {"accuracy": -float("inf"), "f1": -float("inf"), "kappa": -float("inf")}

        # Reinitialize model and optimizer
        self.model = self.ModelClass(**self.model_args).to(self.device)
        self.optimizer = self.OptimizerClass(self.model.parameters(), **self.opt_args)
        self.scheduler = self.SchedulerClass(self.optimizer, **self.sched_args)
        self.criterion = self.LossClass(**self.loss_args)

        self.early_stopper = early_stopping.EarlyStopping(
            patience=self.patience, save_dir=self.model_path, logger=log_message
        )

        print(f"Training fold {fold + 1}/{self.k_folds}...")
        log_message(f"Training fold {fold + 1}/{self.k_folds}...")

    def load_checkpoint(self, fold):
        model_path = os.path.join(self.model_path, f"model_fold_{self.fold + 1}.pth")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.best_val_acc = checkpoint.get("best_val_accuracy", 0.0)
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            self.train_accuracies = checkpoint.get("train_accuracies", [])
            self.val_accuracies = checkpoint.get("val_accuracies", [])
            print(f"Loaded checkpoint from {model_path}, resuming from epoch {self.start_epoch}")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint for fold {fold + 1}: {e}")
            self.start_epoch = 0
            return False
