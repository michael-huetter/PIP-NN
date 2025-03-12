"""
Example input file hon how to train a PIP-NN model
"""

import numpy as np
import matplotlib.pyplot as plt 
import dill

import torch
import torch.nn as nn

from dataset import PIP_DataModule
from pip_nn import PIP_NN
from molecule import molecule


if __name__ == "__main__":

    my_mol = molecule(["O", "H", "H"], num_estates=1)
    my_mol.load_data("sample_data/h2o.pipnn")          

    # load data for training
    p_ij = my_mol.get_pij()
    E = my_mol.get_energy()

    # prepare data for training
    # construct pips
    G1 = lambda p: (p[:, 0] + p[:, 1]) / 2
    G2 = lambda p: (p[:, 0] * p[:, 1])
    G3 = lambda p: p[:, 2]
    my_mol.set_G([G1, G2, G3])

    # serialize the molecule object
    with open('my_mol.pkl', 'wb') as f:
        dill.dump(my_mol, f)

    X = np.array([G1(p_ij), G2(p_ij), G3(p_ij)]).T
    Y0 = E[:, 0, 1]
    Y = np.array([Y0]).T

    num_models = 1
    train_loader, valid_loader = PIP_DataModule(X, Y, batch_size=10).get_loaders()

    for i in range(num_models):
        print("\033[92m" + f"Training model {i}" + "\033[0m")
        model = PIP_NN(
                      m=X.shape[1], 
                      n=Y.shape[1],
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      epochs=300,
                      loss_fn=nn.MSELoss(),
                      optimizer=torch.optim.Adam, 
                      optimizer_params={"lr": 0.01},
                      scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                      scheduler_params={"mode": "min"},
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      verbose=False,
                      interactive=True,
                      model_idx=i
                      )  
        model.print_summary()
        model.train_nn()
