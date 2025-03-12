import sys
import os
import warnings
import numpy as np

def warning(message, category, filename, lineno, file=None, line=None):
    red = "\033[91m"
    reset = "\033[0m"
    formatted_message = f"{red}{category.__name__}: {message}{reset}"
    print(formatted_message, file=file if file is not None else sys.stderr)
warnings.showwarning = warning

class molecule:
    def __init__(self, nuc, num_estates):
        self.nuc = nuc
        self.num_estates = num_estates
        self.pos = np.zeros((len(nuc), 3))
        self.path = None
        self.pos_trace = []
        self.energy_trace = []
        self.rij_trace = []
        self.rij_idx = []
        self.G = []

        if nuc is None:
            warnings.warn("No atoms in the molecule")
        if num_estates < 1:
            warnings.warn("Number of states must be at least 1")

    # getters
    def get_pij(self):
        return np.exp(-np.array(self.rij_trace))
    def get_energy(self):
        return self.energy_trace
    def get_rij(self):
        return self.rij_trace

    def _compute_rij(self):
        print("Computing r_ij ...")
        for i in range(len(self.pos_trace)):
            pos = self.pos_trace[i]
            rij = []
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    if i < j:
                        if (i, j) not in self.rij_idx:
                            self.rij_idx.append((i, j))
                        rij.append(np.linalg.norm(np.array(pos[i][1:]) - np.array(pos[j][1:])))
            self.rij_trace.append(rij)
        print("r_ij computed successfully")
        print(f"Found r_ij index: {self.rij_idx}")
        print(self.nuc)

    
    def load_data(self, path):
        self.path = path
        print("Loading data from", path, "...")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = f.readlines()
            if len(data) == 0:
                warnings.warn(f"Empty file {path}")
        else:
            warnings.warn(f"File {path} does not exist")
        n_at, n_estat = data[0].split()
        n_at = int(n_at)
        n_estat = int(n_estat)
        if n_at != len(self.nuc):
            warnings.warn("Number of atoms in the dataset does not match the number of atoms in the molecule")
        if n_estat != self.num_estates:
            warnings.warn("Number of states in the dataset does not match the number of states in the molecule")
        for i in range(1, len(data)):
            tokens = data[i].split()
            positions = [(tokens[i], float(tokens[i+1]), float(tokens[i+2]), float(tokens[i+3])) 
             for i in range(0, n_at * 4, 4)]
            energies = [(int(tokens[i]), float(tokens[i+1])) 
            for i in range(n_at * 4, n_at * 4 + n_estat * 2, 2)]
            self.pos_trace.append(positions)
            self.energy_trace.append(energies)
        self.energy_trace = np.array(self.energy_trace)
        print("Data loaded successfully")
        self._compute_rij()

    def set_G(self, G):
        self.G = G

    def eval_point(self, pos):
        r_ij = []
        n_atoms = pos.shape[0]
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                d = np.linalg.norm(pos[i] - pos[j])
                r_ij.append(d)
        r_ij = np.array(r_ij)
        p_ij = np.exp(-r_ij)
        p_ij = p_ij.reshape(1, -1)
        
        # If no polynomial functions were set, simply return p_ij
        if not self.G:
            return p_ij
        
        # Evaluate each permutation invariant polynomial on p_ij
        evaluated_polys = []
        for poly in self.G:
            evaluated_polys.append(poly(p_ij))
        return evaluated_polys
