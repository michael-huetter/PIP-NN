import warnings
import sys

import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

#-----------------------------------------------------
def warning(message, category, filename, lineno, file=None, line=None):
    red = "\033[91m"
    reset = "\033[0m"
    formatted_message = f"{red}{category.__name__}: {message}{reset}"
    print(formatted_message, file=file if file is not None else sys.stderr)
warnings.showwarning = warning
#-----------------------------------------------------

class PIP_Dataset(Dataset):
    def __init__(self, pip: torch.Tensor, E: torch.Tensor):
        super(PIP_Dataset, self).__init__()
        self.pip = pip.float()
        self.E = E.float()
        
    def __getitem__(self, i):
        return self.pip[i], self.E[i]

    def __len__(self):
        return self.pip.shape[0]

def scale(t: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor) -> torch.Tensor:
    return 2 * (t - t_min) / (t_max - t_min) - 1

def inverse_scale(t: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor):
    return (t + 1) * (t_max - t_min) * 0.5 + t_min

class PIP_DataModule:
    def __init__(self, X: np.ndarray, Y: np.ndarray, batch_size: int, 
                 test_size: float = 0.2, random_state: int = 15, 
                 verbose: bool = False, retrain: bool = False,
                 old_data_percentage: float = 1.0):
        """
        Parameters:
            X (np.ndarray): Input features.
            Y (np.ndarray): Output targets.
            batch_size (int): Batch size for the DataLoaders.
            test_size (float): Proportion of the data to be used for testing/validation.
            random_state (int): Random seed for reproducibility.
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.retrain = retrain
        self.old_data_percentage = old_data_percentage

        if not X.shape[0] == Y.shape[0]:
            warnings.warn("X and Y must have the same number of samples.")
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            warnings.warn("X and Y must be numpy arrays.")

    def prepare_data(self):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=self.random_state)
        
        if self.retrain:
            X_train_old = np.loadtxt("X_train.txt")
            X_test_old = np.loadtxt("X_valid.txt")
            Y_train_old = np.loadtxt("Y_train.txt")
            Y_test_old = np.loadtxt("Y_valid.txt")
            pct = self.old_data_percentage
            num_train_old = X_train_old.shape[0]
            num_sample_train = int(pct * num_train_old)
            indices_train = np.random.choice(num_train_old, size=num_sample_train, replace=False)
            X_train_old_sampled = X_train_old[indices_train]
            Y_train_old_sampled = Y_train_old[indices_train]
            num_test_old = X_test_old.shape[0]
            num_sample_test = int(pct * num_test_old)
            indices_test = np.random.choice(num_test_old, size=num_sample_test, replace=False)
            X_test_old_sampled = X_test_old[indices_test]
            Y_test_old_sampled = Y_test_old[indices_test]
            X_train = np.concatenate((X_train, X_train_old_sampled))
            X_test = np.concatenate((X_test, X_test_old_sampled))
            Y_train = np.concatenate((Y_train, Y_train_old_sampled))
            Y_test = np.concatenate((Y_test, Y_test_old_sampled))


        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

        X_all_tensor = torch.cat((X_train_tensor, X_test_tensor))
        Y_all_tensor = torch.cat((Y_train_tensor, Y_test_tensor))

        X_min_tensor_new = X_all_tensor.min(dim=0).values
        X_max_tensor_new = X_all_tensor.max(dim=0).values
        Y_min_tensor_new = Y_all_tensor.min(dim=0).values
        Y_max_tensor_new = Y_all_tensor.max(dim=0).values

        if self.retrain:
            X_min_tensor_old = torch.from_numpy(np.loadtxt("X_min.txt"))
            X_max_tensor_old = torch.from_numpy(np.loadtxt("X_max.txt"))
            Y_min_tensor_old = torch.from_numpy(np.loadtxt("Y_min.txt"))
            Y_max_tensor_old = torch.from_numpy(np.loadtxt("Y_max.txt"))
            X_min_tensor = torch.min(X_min_tensor_old, X_min_tensor_new)
            X_max_tensor = torch.max(X_max_tensor_old, X_max_tensor_new)
            # is this correct for 2D?
            Y_min_tensor = torch.min(Y_min_tensor_old, Y_min_tensor_new) 
            Y_max_tensor = torch.max(Y_max_tensor_old, Y_max_tensor_new)
        else:
            X_min_tensor = X_min_tensor_new
            X_max_tensor = X_max_tensor_new
            Y_min_tensor = Y_min_tensor_new
            Y_max_tensor = Y_max_tensor_new

        X_scaled_train_tensor = scale(X_train_tensor, X_min_tensor, X_max_tensor)
        X_scaled_test_tensor = scale(X_test_tensor, X_min_tensor, X_max_tensor)
        Y_scaled_train_tensor = scale(Y_train_tensor, Y_min_tensor, Y_max_tensor)
        Y_scaled_test_tensor = scale(Y_test_tensor, Y_min_tensor, Y_max_tensor)

        train_dataset = PIP_Dataset(X_scaled_train_tensor, Y_scaled_train_tensor)
        test_dataset = PIP_Dataset(X_scaled_test_tensor, Y_scaled_test_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        np.savetxt("X_min.txt", X_min_tensor.detach().numpy())
        np.savetxt("X_max.txt", X_max_tensor.detach().numpy())
        np.savetxt("Y_min.txt", Y_min_tensor.detach().numpy())
        np.savetxt("Y_max.txt", Y_max_tensor.detach().numpy())
        np.savetxt("X_train.txt", X_train)
        np.savetxt("X_valid.txt", X_test)
        np.savetxt("Y_train.txt", Y_train)
        np.savetxt("Y_valid.txt", Y_test)
        if self.retrain:
            np.savetxt("X_train.txt", np.concatenate((X_train, X_train_old)))
            np.savetxt("X_valid.txt", np.concatenate((X_test, X_test_old)))
            np.savetxt("Y_train.txt", np.concatenate((Y_train, Y_train_old)))
            np.savetxt("Y_valid.txt", np.concatenate((Y_test, Y_test_old)))

        if self.verbose:
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"Y_train shape: {Y_train.shape}")
            print(f"Y_test shape: {Y_test.shape}")
            print(f"X_min: {X_min_tensor.detach().numpy()}")
            print(f"X_max: {X_max_tensor.detach().numpy()}")
            print(f"Y_min: {Y_min_tensor.detach().numpy()}")
            print(f"Y_max: {Y_max_tensor.detach().numpy()}")
            if self.retrain:
                print("Retraining model.")
                print(f"X_min_old: {X_min_tensor_old.detach().numpy()}")
                print(f"X_min_new: {X_min_tensor_new.detach().numpy()}")
                print(f"X_max_old: {X_max_tensor_old.detach().numpy()}")
                print(f"X_max_new: {X_max_tensor_new.detach().numpy()}")
                print(f"Y_min_old: {Y_min_tensor_old.detach().numpy()}")
                print(f"Y_min_new: {Y_min_tensor_new.detach().numpy()}")
                print(f"Y_max_old: {Y_max_tensor_old.detach().numpy()}")
                print(f"Y_max_new: {Y_max_tensor_new.detach().numpy()}")

    def get_loaders(self):
        """
        Prepares the data (if not already done) and returns the train and test DataLoaders.
        """
        if not hasattr(self, "train_loader") or not hasattr(self, "test_loader"):
            self.prepare_data()
        return self.train_loader, self.test_loader
