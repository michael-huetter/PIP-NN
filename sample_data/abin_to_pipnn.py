#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

class MDConverter:
    def __init__(self, energies_file, movie_file, output_file):
        """
        energies_file: path to energies.dat file (with header lines starting with '#')
        movie_file: path to movie.xyz file (multiple frames in xyz format)
        output_file: path for the converted output file
        """
        self.energies_file = energies_file
        self.movie_file = movie_file
        self.output_file = output_file
        self.energies = []  # List to store ground-state electronic energies for each frame
        self.frames = []    # List to store frames; each frame is a list of (symbol, x, y, z) tuples

    def read_energies(self):
        """Read energies.dat and extract the E-potential (second column) from each non-header line."""
        with open(self.energies_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    e_potential = float(parts[1])
                    self.energies.append(e_potential)
                except ValueError:
                    continue

    def read_movie(self):
        """
        Read the movie.xyz file.
        Assumes each frame starts with a line containing the number of atoms,
        followed by a comment line, and then that many lines of atomic coordinates.
        """
        with open(self.movie_file, 'r') as f:
            lines = f.readlines()
        
        idx = 0
        while idx < len(lines):
            # Skip any empty lines
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if idx >= len(lines):
                break
            try:
                natoms = int(lines[idx].strip())
            except ValueError:
                raise ValueError(f"Expected an integer for number of atoms at line {idx+1}")
            idx += 1
            # Skip comment line
            if idx < len(lines):
                idx += 1
            frame = []
            for i in range(natoms):
                if idx >= len(lines):
                    raise ValueError("Unexpected end of file while reading atom coordinates.")
                parts = lines[idx].strip().split()
                if len(parts) < 4:
                    raise ValueError(f"Expected at least 4 columns (symbol and 3 coords) at line {idx+1}")
                symbol = parts[0]
                try:
                    x, y, z = map(float, parts[1:4])
                except ValueError:
                    raise ValueError(f"Could not convert coordinates to float at line {idx+1}")
                frame.append((symbol, x, y, z))
                idx += 1
            self.frames.append(frame)
    
    def convert(self):
        """Convert the energies and movie files to the desired output format."""
        self.read_energies()
        self.read_movie()
        
        n_frames = len(self.frames)
        if n_frames != len(self.energies):
            raise ValueError("Mismatch in number of frames between energies and movie files.")
        
        natoms = len(self.frames[0])
        # Only one electronic state (ground state)
        n_states = 1
        
        output_lines = []
        # Write header: number of atoms and electronic states (1)
        header = f"{natoms} {n_states}"
        output_lines.append(header)
        
        for i in range(n_frames):
            frame = self.frames[i]
            energy = self.energies[i]
            tokens = []
            # Write coordinates for all atoms except the last one normally.
            for atom in frame[:-1]:
                symbol, x, y, z = atom
                tokens.append(f"{symbol} {x:.3f} {y:.3f} {z:.3f}")
            # For the last atom, append the ground state info: state index 0 and the energy.
            symbol, x, y, z = frame[-1]
            atom_info = f"{symbol} {x:.3f} {y:.3f} {z:.3f}"
            state_info = f"0 {energy:.3f}"
            tokens.append(f"{atom_info} {state_info}")
            
            output_lines.append(" ".join(tokens))
        
        with open(self.output_file, 'w') as f:
            for line in output_lines:
                f.write(line + "\n")
        
        print(f"Conversion complete. {n_frames} frames written to '{self.output_file}'.")

    # --- Methods for computing internal coordinates ---
    @staticmethod
    def compute_distance(atom1, atom2):
        """Return the Euclidean distance between two atoms."""
        _, x1, y1, z1 = atom1
        _, x2, y2, z2 = atom2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    @staticmethod
    def compute_angle(atom_i, atom_j, atom_k):
        """
        Compute the angle (in degrees) with vertex at atom_j, given three atoms.
        The angle is defined between vectors (atom_j -> atom_i) and (atom_j -> atom_k).
        """
        _, xi, yi, zi = atom_i
        _, xj, yj, zj = atom_j
        _, xk, yk, zk = atom_k
        v1 = (xi - xj, yi - yj, zi - zj)
        v2 = (xk - xj, yk - yj, zk - zj)
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        norm1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        norm2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if norm1 * norm2 == 0:
            return 0.0
        cos_angle = dot / (norm1 * norm2)
        # Clamp cos_angle to avoid numerical issues
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))
    
    @staticmethod
    def compute_dihedral(atom1, atom2, atom3, atom4):
        """
        Compute the dihedral angle (in degrees) defined by four atoms.
        Uses the standard definition via cross products.
        """
        p0 = np.array(atom1[1:])
        p1 = np.array(atom2[1:])
        p2 = np.array(atom3[1:])
        p3 = np.array(atom4[1:])
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        # Normalize b1 so that it does not influence magnitude of vector rejections
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return math.degrees(math.atan2(y, x))
    
    def compute_internal_coordinates(self):
        """
        Automatically compute a dictionary of internal coordinates over all frames.
        The keys are strings describing the coordinate (e.g., "Distance: O1-H2" or "Angle: H1-O1-H2"),
        and the values are lists of computed values (one per frame).
        Computes:
         - All pairwise distances (for i < j)
         - All angles for triplets with i < j < k (angle at j)
         - All dihedral angles for quadruplets with i < j < k < l (if number of atoms >= 4)
        """
        if not self.frames:
            raise ValueError("No frames available. Ensure movie file is read before computing internal coordinates.")
        
        n_frames = len(self.frames)
        n_atoms = len(self.frames[0])
        coords_dict = {}

        # --- Compute distances ---
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                label = f"Distance: {self.frames[0][i][0]}{i+1}-{self.frames[0][j][0]}{j+1}"
                values = []
                for frame in self.frames:
                    d = self.compute_distance(frame[i], frame[j])
                    values.append(d)
                coords_dict[label] = values
        
        # --- Compute angles (using triplets with i < j < k, angle at atom j) ---
        if n_atoms >= 3:
            for j in range(1, n_atoms - 1):
                for i in range(j):
                    for k in range(j + 1, n_atoms):
                        label = f"Angle: {self.frames[0][i][0]}{i+1}-{self.frames[0][j][0]}{j+1}-{self.frames[0][k][0]}{k+1}"
                        values = []
                        for frame in self.frames:
                            a = self.compute_angle(frame[i], frame[j], frame[k])
                            values.append(a)
                        coords_dict[label] = values

        # --- Compute dihedral angles (for quadruplets, if available) ---
        if n_atoms >= 4:
            for i in range(n_atoms - 3):
                for j in range(i + 1, n_atoms - 2):
                    for k in range(j + 1, n_atoms - 1):
                        for l in range(k + 1, n_atoms):
                            label = f"Dihedral: {self.frames[0][i][0]}{i+1}-{self.frames[0][j][0]}{j+1}-{self.frames[0][k][0]}{k+1}-{self.frames[0][l][0]}{l+1}"
                            values = []
                            for frame in self.frames:
                                d = self.compute_dihedral(frame[i], frame[j], frame[k], frame[l])
                                values.append(d)
                            coords_dict[label] = values

        return coords_dict

    def visualize_pes(self):
        """
        Visualize the PES (Electronic Energy vs. Internal Coordinate).
        Computes internal coordinates automatically from the movie file,
        then opens a matplotlib window with a radio-button widget listing all available
        internal coordinates. Selecting one updates the scatter plot.
        """
        # Ensure energies and frames are read (or already computed)
        if not self.energies or not self.frames:
            self.read_energies()
            self.read_movie()
        coords_dict = self.compute_internal_coordinates()
        if not coords_dict:
            raise ValueError("No internal coordinates computed.")

        # Use the first coordinate as default.
        keys = list(coords_dict.keys())
        current_key = keys[0]
        xdata = coords_dict[current_key]
        ydata = self.energies
        ydata = np.array(ydata)
        ydata = (ydata - np.min(ydata))*27.2114  # Convert to eV

        # Create the figure and scatter plot.
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.3)  # leave space for the widget
        sc = ax.scatter(xdata, ydata)
        ax.set_xlabel(current_key)
        ax.set_ylabel("Ground State Energy")
        ax.set_title(f"PES: {current_key} vs Energy")

        # Create an axes for the radio buttons.
        rax = plt.axes([0.05, 0.4, 0.2, 0.5])
        radio = RadioButtons(rax, keys, active=0)

        def update(label):
            ax.cla()  # clear current axes
            ax.scatter(coords_dict[label], ydata)
            ax.set_xlabel(label)
            ax.set_ylabel("Energy in eV") 
            ax.set_title(f"PES: {label} vs Energy")
            fig.canvas.draw_idle()

        radio.on_clicked(update)
        plt.show()


if __name__ == "__main__":
    converter = MDConverter("energies.dat", "movie.xyz", "h2o.pipnn")
    
    converter.convert()
    
    # Then, visualize the PES based on internal coordinates.
    # A matplotlib window will open with a radio-button widget to choose the coordinate.
    converter.visualize_pes()
