"""Rosetta energy calculation."""

import os
import sys
import contextlib
import pyrosetta
import numpy as np
from typing import Dict, List, Any
from io import StringIO
pyrosetta.init()

@contextlib.contextmanager
def suppress_rosetta_output():
    """Context manager to suppress PyRosetta output and warnings."""
    import warnings
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    old_tracer_level = None
    try:
        from pyrosetta.rosetta.basic import Tracer
        old_tracer_level = Tracer.get_all_channels_string()
        Tracer.set_ios_hook(None, None, 100)
    except:
        pass
    
    try:
        warnings.filterwarnings('ignore')
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        warnings.resetwarnings()
        
        if old_tracer_level is not None:
            try:
                Tracer.set_ios_hook(sys.stdout, sys.stderr, 0)
            except:
                pass

def calculate_per_residue_energies(pose, scorefxn) -> Dict[int, Dict[str, float]]:
    """
    Calculate per-residue energy contributions.
    
    Args:
        pose: PyRosetta pose object
        scorefxn: PyRosetta score function
        
    Returns:
        Dictionary mapping residue positions (0-indexed) to energy terms
    """
    per_residue_data = {}
    
    try:
        pose.energies().clear()
        scorefxn(pose)
        
        for i in range(1, pose.total_residue() + 1):
            residue = pose.residue(i)
            residue_name = residue.name1()
            
            residue_energies = pose.energies().residue_total_energies(i)
            
            vdw_atr = residue_energies[pyrosetta.rosetta.core.scoring.fa_atr]
            vdw_rep = residue_energies[pyrosetta.rosetta.core.scoring.fa_rep]
            hbond = (residue_energies[pyrosetta.rosetta.core.scoring.hbond_sr_bb] +
                     residue_energies[pyrosetta.rosetta.core.scoring.hbond_lr_bb] +
                     residue_energies[pyrosetta.rosetta.core.scoring.hbond_bb_sc] +
                     residue_energies[pyrosetta.rosetta.core.scoring.hbond_sc])
            total = residue_energies.sum()
            
            per_residue_data[i - 1] = {
                'residue': residue_name,
                'total_energy': total,
                'vdw_atr': vdw_atr,
                'vdw_rep': vdw_rep,
                'hbond': hbond,
                'rosetta_position': i
            }
            
    except Exception as e:
        print(f"Error calculating per-residue energies: {str(e)}")
        
    return per_residue_data

def calculate_rosetta_energies(pdb_path: str, per_residue: bool = False) -> Dict[str, Any]:
    """
    Calculate Rosetta energies for a protein structure.
    
    Args:
        pdb_path: Path to the PDB file
        per_residue: If True, calculate per-residue energies
        
    Returns:
        Dictionary with energy terms or None if calculation fails
    """
    if not pdb_path or not os.path.exists(pdb_path):
        print(f"Warning: PDB file not found at {pdb_path}")
        return None
    
    if os.path.getsize(pdb_path) == 0:
        print(f"Warning: PDB file is empty: {pdb_path}")
        return None
    
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
            atom_lines = [line for line in lines if line.startswith('ATOM')]
            if len(atom_lines) == 0:
                print(f"Warning: PDB file contains no ATOM records: {pdb_path}")
                return None
    except Exception as e:
        print(f"Warning: Could not read PDB file {pdb_path}: {e}")
        return None
    
    try:
        with suppress_rosetta_output():
            pose = pyrosetta.pose_from_pdb(pdb_path)
            scorefxn = pyrosetta.get_fa_scorefxn()
            score = scorefxn(pose)
            
            energies = pose.energies().total_energies()
            vdw_atr = energies[pyrosetta.rosetta.core.scoring.fa_atr]
            vdw_rep = energies[pyrosetta.rosetta.core.scoring.fa_rep]
            hbond = (energies[pyrosetta.rosetta.core.scoring.hbond_sr_bb] +
                     energies[pyrosetta.rosetta.core.scoring.hbond_lr_bb] +
                     energies[pyrosetta.rosetta.core.scoring.hbond_bb_sc] +
                     energies[pyrosetta.rosetta.core.scoring.hbond_sc])
            total = energies.sum()
        
        result = {
            'score': score,
            'vdw_atr': vdw_atr,
            'vdw_rep': vdw_rep,
            'hbond': hbond,
            'total_energy': total,
            'full_energies': energies
        }
        
        if per_residue:
            per_residue_energies = calculate_per_residue_energies(pose, scorefxn)
            result['per_residue'] = per_residue_energies
        
        del pose
        del scorefxn
        if 'per_residue_energies' in locals():
            del per_residue_energies
        
        return result
        
    except Exception as e:
        print(f"Error calculating Rosetta energies for {pdb_path}: {str(e)}")
        return None

def mutate_and_calculate_energy(pdb_path, position, new_aa):
    """
    Mutate a residue and calculate the energy change.
    
    Args:
        pdb_path: Path to the PDB file.
        position: Residue position to mutate (0-indexed).
        new_aa: One-letter amino acid code for the new residue.
        
    Returns:
        Dictionary with energy before and after mutation, and the difference.
    """
    try:
        with suppress_rosetta_output():
            pose = pyrosetta.pose_from_file(pdb_path)
            scorefxn = pyrosetta.get_fa_scorefxn()
        
        energy_before = scorefxn(pose)
        
        ros_pos = position + 1
        
        original_aa = pose.residue(ros_pos).name1()
        
        mutated_pose = pyrosetta.Pose()
        mutated_pose.assign(pose)
        pyrosetta.toolbox.mutants.mutate_residue(mutated_pose, ros_pos, new_aa)
        
        min_mover = pyrosetta.MinMover()
        mm = pyrosetta.MoveMap()
        mm.set_bb(False)
        mm.set_chi(True)
        mm.set_chi(ros_pos, True)
        for i in range(1, pose.total_residue() + 1):
            if pose.residue(ros_pos).xyz("CA").distance(pose.residue(i).xyz("CA")) < 8.0:
                mm.set_chi(i, True)
        min_mover.movemap(mm)
        min_mover.score_function(scorefxn)
        min_mover.apply(mutated_pose)
        
        energy_after = scorefxn(mutated_pose)
        
        return {
            'original_aa': original_aa,
            'new_aa': new_aa,
            'energy_before': energy_before,
            'energy_after': energy_after,
            'energy_diff': energy_after - energy_before
        }
    
    except Exception as e:
        print(f"Error in mutation: {str(e)}")
        return None

def get_energy_guidance(energies: Dict[str, Any]) -> str:
    """
    Generate guidance text based on energy calculations.
    
    Args:
        energies: Dictionary with energy terms from calculate_rosetta_energies
        
    Returns:
        String with guidance for improving energy
    """
    if not energies:
        return ""
    
    guidance = "Energy considerations:\n"
    
    if energies['vdw_rep'] > 10:
        guidance += "- High repulsive energy detected. Consider mutations that reduce steric clashes.\n"
    
    if energies['hbond'] > -5:
        guidance += "- Low hydrogen bonding energy. Consider mutations that could form additional H-bonds.\n"
    
    if energies['vdw_atr'] > -50:
        guidance += "- Weak attractive forces. Consider mutations that improve packing.\n"
    
    return guidance

def analyze_pdb_files(pdb_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple PDB files and return their energy scores.
    
    Args:
        pdb_files: List of paths to PDB files
        
    Returns:
        Dictionary mapping file paths to energy dictionaries
    """
    results = {}
    
    for pdb_file in pdb_files:
        if os.path.exists(pdb_file):
            energies = calculate_rosetta_energies(pdb_file)
            if energies:
                results[pdb_file] = energies
                
                print(f"Score for {pdb_file}: {energies['score']}")
                print(f"Van der Waals attractive (fa_atr): {energies['vdw_atr']:.3f}")
                print(f"Van der Waals repulsive (fa_rep): {energies['vdw_rep']:.3f}")
                print(f"Hydrogen bonding (total): {energies['hbond']:.3f}")
                print(f"Total Rosetta Energy: {energies['total_energy']:.3f}\n")
        else:
            print(f"File not found: {pdb_file}")
    
    return results 