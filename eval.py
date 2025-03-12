import numpy as np
import dill
import torch

from dataset import PIP_DataModule
from pip_nn import PIP_NN
from molecule import molecule

from matplotlib import pyplot as plt

# -----------------------------------------------------
model = PIP_NN.load_model("pip_nn_0.pth", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
with open('my_mol.pkl', 'rb') as f:
    my_mol = dill.load(f)
# -----------------------------------------------------
pos_sample = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.96],
        [0.0, 0.75, -0.36]
    ])
G = my_mol.eval_point(pos_sample)
G = np.array(G).T
# -----------------------------------------------------
Y_pred = model.eval_nn(G)
print(Y_pred)
