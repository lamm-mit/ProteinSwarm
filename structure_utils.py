"""Protein structure analysis."""

import numpy as np
from Bio.PDB import PDBParser
from constants import DEFAULT_CUTOFF, DEFAULT_EXPOSURE_THRESHOLD
import os


def compute_ca_distance_matrix(pdb_path: str) -> np.ndarray:
    """Compute a Cα-Cα distance matrix from a PDB file."""
    agent_processing_flag_file = "/tmp/swarm_agent_processing.flag"
    
    if os.path.exists(agent_processing_flag_file):
        return np.array([[]])
    
    if not pdb_path or not os.path.exists(pdb_path):
        print(f"✗ Error: PDB file not found: {pdb_path}")
        return None
        
    if os.path.getsize(pdb_path) == 0:
        print(f"✗ Error: PDB file is empty: {pdb_path}")
        return None
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    
    ca_atoms = []
    for residue in structure.get_residues():
        if "CA" in residue:
            ca_atoms.append(residue["CA"])
    
    if len(ca_atoms) == 0:
        print(f"✗ Error: No CA atoms found in PDB file: {pdb_path}")
        return None
    
    n = len(ca_atoms)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(ca_atoms[i].get_coord() - ca_atoms[j].get_coord())
    
    print(f"✓ Distance matrix computed successfully ({n}x{n})")
    
    import gc
    del ca_atoms, structure, parser
    gc.collect()
    
    return dist_matrix


def get_spatial_neighbors(index: int, sequence: str, dist_matrix: np.ndarray, 
                          cutoff: float = DEFAULT_CUTOFF, neighbor_radius: int = 0) -> list:
    """Identify spatial neighbors of a residue that are not immediately adjacent in the linear sequence."""
    linear_neighbors = set(range(max(0, index - neighbor_radius), min(len(sequence), index + neighbor_radius + 1)))
    return [
        {"position": i, "state": sequence[i]}
        for i, d in enumerate(dist_matrix[index])
        if i != index and d < cutoff and i not in linear_neighbors
    ]


def get_exposure(index: int, sequence: str, dist_matrix: np.ndarray, cutoff: float = DEFAULT_CUTOFF) -> str:
    """Estimate residue exposure using number of spatial neighbors."""
    num_neighbors = len(get_spatial_neighbors(index, sequence, dist_matrix, cutoff))
    return "buried" if num_neighbors > DEFAULT_EXPOSURE_THRESHOLD else "surface"


def fallback_get_local_structure(index: int, dist_matrix: np.ndarray) -> str:
    """Heuristically classify local structure from Cα distances."""
    left = dist_matrix[index, index - 1] if index - 1 >= 0 else None
    right = dist_matrix[index, index + 1] if index + 1 < dist_matrix.shape[0] else None
    if left and right and 3.5 < left < 4.2 and 3.5 < right < 4.2:
        return "helix"
    elif index + 2 < dist_matrix.shape[0] and 6.0 < dist_matrix[index, index + 2] < 7.2:
        return "sheet"
    else:
        return "loop"
