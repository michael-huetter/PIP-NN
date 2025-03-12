import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from time import time
from datetime import timedelta

#-----------------------------------------------------
def warning(message, category, filename, lineno, file=None, line=None):
    red = "\033[91m"
    reset = "\033[0m"
    formatted_message = f"{red}{category.__name__}: {message}{reset}"
    print(formatted_message, file=file if file is not None else sys.stderr)
warnings.showwarning = warning
#-----------------------------------------------------

class PIP_NN(nn.Module):
    def __init__(self, m, n, 
                 train_loader, 
                 valid_loader=None, 
                 epochs=100, 
                 loss_fn=None,
                 optimizer=torch.optim.Adam, 
                 optimizer_params=None,
                 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                 scheduler_params=None,
                 device=None,
                 verbose=True,
                 interactive=False,
                 model_idx=0):
        """
        Parameters:
            m (int): Input dimension.
            n (int): Output dimension.
            train_loader (DataLoader): Dataloader for training data.
            valid_loader (DataLoader): Dataloader for validation data (default: train_loader).
            epochs (int): Number of epochs to train.
            loss_fn (callable): Loss function (default: nn.MSELoss()).
            optimizer (torch.optim.Optimizer): Optimizer class (default: torch.optim.Adam).
            optimizer_params (dict): Parameters for the optimizer (e.g. {"lr": 0.001}).
            scheduler (torch.optim.lr_scheduler): Scheduler class (default: ReduceLROnPlateau).
            scheduler_params (dict): Parameters for the scheduler.
            device (torch.device): Device to train on (default: "cuda" if available else "cpu").
            verbose (bool): Whether to print training progress.
            interactive (bool): Whether to show an interactive plot.
            model_idx (int): Index used for saving the model.
        """
        super(PIP_NN, self).__init__()
        self.m = m
        self.n = n
        self.train_loader = train_loader
        self.valid_loader = valid_loader if valid_loader is not None else train_loader
        self.epochs = epochs
        self.verbose = verbose
        self.interactive = interactive
        self.model_idx = model_idx
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss() 
        # to be initialized in the load_model method
        self.X_min = None
        self.X_max = None
        self.Y_min = None
        self.Y_max = None           
        
        self.layer_stack = nn.Sequential(
            nn.Linear(m, 10),
            nn.Tanh(),
            nn.Linear(10, 50),
            nn.Tanh(),
            nn.Linear(50, n)
        )

        if optimizer_params is None:
            optimizer_params = {"lr": 0.001}
        self.optimizer = optimizer(self.parameters(), **optimizer_params)
        
        if scheduler_params is None:
            scheduler_params = {}
        self.scheduler = scheduler(self.optimizer, **scheduler_params)
        
        self.to(self.device)
        
        # Variable used for interactive mode to stop training early
        self.stop_loop = False

    def print_summary(self):
        print(f"Number of input features: {self.m}")
        print(f"Number of output features: {self.n}")
        print(f"Training data: {len(self.train_loader.dataset)} samples")
        print(f"Validation data: {len(self.valid_loader.dataset)} samples")
        print(f"Epochs: {self.epochs}")
        print(f"Training on device: {self.device}")
        print(f"Loss function: {self.loss_fn}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Scheduler: {self.scheduler}")

    @classmethod
    def load_model(cls, path, device=None):
        model = torch.load(path, map_location=device)
        model.X_min = torch.from_numpy(np.loadtxt("X_min.txt"))
        model.X_max = torch.from_numpy(np.loadtxt("X_max.txt"))
        model.Y_min = torch.from_numpy(np.loadtxt("Y_min.txt"))
        model.Y_max = torch.from_numpy(np.loadtxt("Y_max.txt"))
        if device is not None:
            model.to(device)
        print(f"Model loaded from {path}")
        return model

    def forward(self, x):
        return self.layer_stack(x)
    
    def _train_one_epoch(self):
        self.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self(X)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def _validate(self):
        self.eval()
        total_loss = 0.0
        n_batches = len(self.valid_loader)
        with torch.no_grad():
            for X, y in self.valid_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self(X)
                total_loss += self.loss_fn(y_pred, y).item()
        return total_loss / n_batches if n_batches > 0 else 0
    
    def _on_button_click(self, event):
        """Callback to set stop_loop flag when the interactive stop button is pressed."""
        self.stop_loop = True

    def train_nn(self):
        self.stop_loop = False
        writer = SummaryWriter("logs/" + str(int(time())))
        start_time = time()
        
        # Set up interactive plotting if enabled
        if self.interactive:
            plt.ion()
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)
            sc = ax.scatter([], [], c=[], cmap='coolwarm', vmin=0, vmax=1)
            ax.set_xlim(0, self.epochs)
            ax.set_ylim(0, 0.1)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Progress")
            # Dummy points for legend
            dummy_train = ax.scatter([], [], color='blue', label='Training Loss')
            dummy_valid = ax.scatter([], [], color='red', label='Validation Loss')
            ax.legend()
            button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
            stop_button = Button(button_ax, 'Stop')
            stop_button.on_clicked(self._on_button_click)
            points = []
            point_types = []
        
        # Training loop
        for epoch in range(self.epochs):
            if self.interactive and self.stop_loop:
                print("Training stopped by button press.")
                break
            epoch_start = time()
            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", current_lr, epoch)
            
            train_loss = self._train_one_epoch()
            writer.add_scalar("train_loss", train_loss, epoch)
            train_end = time()
            
            valid_loss = self._validate()
            writer.add_scalar("valid_loss", valid_loss, epoch)
            writer.flush()
            valid_end = time()
            
            self.scheduler.step(valid_loss)
            epoch_end = time()
            
            if self.verbose:
                print(f"Epoch {epoch + 1}")
                print("-" * 80)
                print(f"Learning rate: {current_lr:.7f}")
                print(f"Epoch time: {timedelta(seconds=(epoch_end - epoch_start))}")
                print(f"Train time: {timedelta(seconds=(train_end - epoch_start))}")
                print(f"Valid time: {timedelta(seconds=(valid_end - train_end))}")
                print(f"Train loss: {train_loss:.6f}")
                print(f"Valid loss: {valid_loss:.6f}\n")
            
            if self.interactive:
                # Record and update the plot with new training and validation losses
                points.append([epoch, float(train_loss)])
                point_types.append(0)  # 0 for training
                points.append([epoch, float(valid_loss)])
                point_types.append(1)  # 1 for validation
                points_arr = np.array(points)
                types_arr = np.array(point_types)
                sc.set_offsets(points_arr)
                sc.set_array(types_arr)
                plt.pause(0.05)
                
        if self.interactive:
            plt.ioff()
            plt.show()
            
        end_time = time()
        print("Done!")
        print(f"Total time: {timedelta(seconds=(end_time - start_time))}")
        # Save the model to file
        save_path = f"pip_nn_{self.model_idx}.pth"
        torch.save(self, save_path)
        print(f"Model saved to {save_path}")

    def scale(self, X):
        return 2 * (X - self.X_min) / (self.X_max - self.X_min) - 1
    
    def inverse_scale(self, X):
        return (X + 1) * (self.Y_max - self.Y_min) * 0.5 + self.Y_min
    
    def eval_nn(self, X):
        if X is not np.ndarray:
            warnings.warn("Input must be a numpy array.")
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_scaled_tensor = self.scale(X_tensor).float()
        with torch.no_grad():
            predictions = self(X_scaled_tensor)
        return self.inverse_scale(predictions).detach().numpy()
    
    def retrain_nn(self, new_train_loader, new_valid_loader=None, update_scaling=True, additional_epochs=None, finetune_lr=None):
        """
        Fine-tunes the model using new training data points from an extrapolation region,
        reusing the training and plotting code from train_nn().

        Parameters:
            new_train_loader (DataLoader): DataLoader for the new training data.
            new_valid_loader (DataLoader, optional): DataLoader for validation data.
                If not provided, new_train_loader is used.
            update_scaling (bool): If True, update the scaling parameters (X_min, X_max, Y_min, Y_max)
                to reflect the new data range combined with the old.
            additional_epochs (int, optional): Number of additional epochs to train.
                If None, self.epochs is used.
        """

        warnings.warn("Note that the model is being retrained with the combined old and new data.")
        warnings.warn("This may lead to overfitting if the new data is not representative of the extrapolation region.")
        warnings.warn("min and max values for X and Y are being updated to reflect the new data range.")
        
        original_train_loader = self.train_loader
        original_valid_loader = self.valid_loader
        original_epochs = self.epochs

        if new_valid_loader is None:
            new_valid_loader = new_train_loader

        if update_scaling:
            self.Y_min = torch.from_numpy(np.loadtxt("Y_min.txt"))
            self.Y_max = torch.from_numpy(np.loadtxt("Y_max.txt"))
            self.X_min = torch.from_numpy(np.loadtxt("X_min.txt"))
            self.X_max = torch.from_numpy(np.loadtxt("X_max.txt"))
        
        if finetune_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = finetune_lr
            print(f"Learning rate set to {finetune_lr} for fine-tuning.")

        self.train_loader = new_train_loader
        self.valid_loader = new_valid_loader
        if additional_epochs is not None:
            self.epochs = additional_epochs

        print(f"Starting retraining for {self.epochs} epochs using new data...")
        self.train_nn()

        self.train_loader = original_train_loader
        self.valid_loader = original_valid_loader
        self.epochs = original_epochs