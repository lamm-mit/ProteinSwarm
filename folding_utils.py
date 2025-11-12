"""Protein folding and visualization."""

import os
import subprocess
import tempfile
import shutil
import py3Dmol
from constants import PYMOL_BINARY


def fold_with_omegafold(fasta_seq, output_dir='omegafold_output'):
    """
    Fold a protein sequence using OmegaFold.
    
    Args:
        fasta_seq: Protein sequence string
        output_dir: Directory to save outputs
        
    Returns:
        Path to generated PDB file
    """
    agent_processing_flag_file = "/tmp/swarm_agent_processing.flag"
    
    if os.path.exists(agent_processing_flag_file):
        return "/dev/null"
    
    os.makedirs(output_dir, exist_ok=True)
    
    fasta_path = os.path.join(output_dir, "input.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">protein\n{fasta_seq}\n")
    
    if not fasta_seq or len(fasta_seq.strip()) == 0:
        print(f"Error: Empty protein sequence provided to OmegaFold")
        return None
    
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_chars = set(fasta_seq.upper()) - valid_aa
    if invalid_chars:
        print(f"Error: Invalid amino acid characters in sequence: {invalid_chars}")
        return None
    
    print(f"Running OmegaFold on sequence of length {len(fasta_seq)}...")
    
    import torch
    
    if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES'):
        cuda_device_id = os.environ.get('CUDA_VISIBLE_DEVICES')
        device = f'cuda:{cuda_device_id}'
        print(f"  🚀 Using GPU: CUDA_VISIBLE_DEVICES={cuda_device_id} -> {device}")
    else:
        device = 'cpu'
        print(f"  💻 Using CPU (GPU not available)")
    
    command = ["omegafold", '--device', device, '--model', '2', fasta_path, output_dir]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        
        if result.returncode != 0:
            print(f"Error: OmegaFold failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Error: OmegaFold timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"Error running OmegaFold: {e}")
        return None
    
    pdb_path = os.path.join(output_dir, "protein.pdb")
    if not os.path.exists(pdb_path):
        print(f"Error: OmegaFold did not generate expected file: {pdb_path}")
        print(f"Output directory contents: {os.listdir(output_dir) if os.path.exists(output_dir) else 'Directory not found'}")
        return None
    
    if os.path.getsize(pdb_path) == 0:
        print(f"Error: Generated PDB file is empty: {pdb_path}")
        return None
    
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
        atom_lines = [line for line in lines if line.startswith('ATOM')]
        if len(atom_lines) == 0:
            print(f"Error: Generated PDB file contains no ATOM records: {pdb_path}")
            return None
    
    print(f"✓ OmegaFold completed successfully: {pdb_path}")
    
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    return pdb_path


def visualize_pdb(pdb_file):
    """
    Visualize a PDB file using py3Dmol in a Jupyter notebook.
    
    Args:
        pdb_file: Path to the PDB file to visualize
    """
    with open(pdb_file, "r") as f:
        pdb_data = f.read()
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    view.show()


def render_protein_with_pymol(pdb_path: str, png_path: str, width: int = 800, height: int = 600):
    """
    Render a protein structure from a PDB file using PyMOL and save the image as PNG.
    Falls back gracefully if PyMOL is not available.
    
    Args:
        pdb_path: Path to the input PDB file
        png_path: Path where the PNG image will be saved
        width: Width of the output image
        height: Height of the output image
        
    Returns:
        True if successful, False otherwise (but doesn't break pipeline)
    """
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file not found: {pdb_path}")
        return False
    
    if not os.path.exists(PYMOL_BINARY):
        print(f"⚠️  PyMOL not found at {PYMOL_BINARY} - skipping visualization")
        print(f"   📁 PDB structure available at: {pdb_path}")
        _create_structure_info_file(pdb_path, png_path)
        return True
    
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    
    abs_pdb_path = os.path.abspath(pdb_path)
    abs_png_path = os.path.abspath(png_path)
    
    temp_dir = tempfile.mkdtemp()
    temp_png = os.path.join(temp_dir, "temp_render.png")

    script = f"""
load {abs_pdb_path}
hide everything
show cartoon
set cartoon_smooth_loops, on
color chainbow
bg_color white
set ray_opaque_background, off
orient
zoom all, 1.2
png {temp_png}, width={width}, height={height}, dpi=300, ray=1
quit
"""
    
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as script_file:
        script_file.write(script)
        script_path = script_file.name

    try:
        result = subprocess.run([
            PYMOL_BINARY, "-cq", script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)

        if os.path.exists(temp_png) and os.path.getsize(temp_png) > 0:
            shutil.copy2(temp_png, abs_png_path)
            print(f"✅ Rendered image saved to {abs_png_path}")
            return True
        else:
            print(f"⚠️  PyMOL execution failed to produce output - creating info file instead")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            _create_structure_info_file(pdb_path, png_path)
            return True
            
    except subprocess.TimeoutExpired:
        print(f"⚠️  PyMOL rendering timed out after 60 seconds - creating info file instead")
        _create_structure_info_file(pdb_path, png_path)
        return True
        
    except FileNotFoundError:
        print(f"⚠️  PyMOL binary not found - skipping visualization")
        _create_structure_info_file(pdb_path, png_path)
        return True
        
    except Exception as e:
        print(f"⚠️  PyMOL rendering failed: {str(e)} - creating info file instead")
        _create_structure_info_file(pdb_path, png_path)
        return True
        
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _create_structure_info_file(pdb_path: str, png_path: str):
    """
    Helper function to create an info file when PyMOL rendering fails.
    """
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    
    info_path = png_path.replace('.png', '_structure_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Protein Structure Information\n")
        f.write(f"============================\n")
        f.write(f"PDB file: {pdb_path}\n")
        f.write(f"Generated: {os.path.getctime(pdb_path) if os.path.exists(pdb_path) else 'Unknown'}\n")
        f.write(f"File size: {os.path.getsize(pdb_path)} bytes\n")
        f.write(f"PyMOL rendering failed or not available\n")
        f.write(f"Use this PDB file with external visualization tools\n")
    
    print(f"📝 Structure info saved: {info_path}")


def log_sequence(log_dir: str, iteration: int, sequence: str, pdb_path: str):
    """
    Save sequence, structure, and rendering image to disk for a design iteration.
    
    Args:
        log_dir: Directory to save the files
        iteration: Iteration number
        sequence: Protein sequence
        pdb_path: Path to the PDB file
    """
    os.makedirs(log_dir, exist_ok=True)

    fasta_path = os.path.join(log_dir, f"seq_iter_{iteration}.fasta")
    with open(fasta_path, "w") as f:
        f.write(sequence)

    final_pdb_path = os.path.join(log_dir, f"struct_iter_{iteration}.pdb")
    if os.path.exists(pdb_path) and pdb_path != final_pdb_path:
        shutil.move(pdb_path, final_pdb_path)
    elif not os.path.exists(final_pdb_path):
        print(f"Warning: PDB file not found at {pdb_path}")
        return

    png_path = os.path.join(log_dir, f"image_iter_{iteration}.png")
    success = render_protein_with_pymol(final_pdb_path, png_path)
    if not success:
        print(f"Warning: Failed to render protein for iteration {iteration}")


def log_sequence_with_status(log_dir: str, iteration: int, sequence: str, pdb_path: str, status: str = "accepted", energy_info: dict = None):
    """
    Save sequence, structure, and rendering image to disk for a design iteration with status tracking.
    
    Args:
        log_dir: Directory to save the files
        iteration: Iteration number
        sequence: Protein sequence
        pdb_path: Path to the PDB file
        status: "accepted", "rejected", or "proposed"
        energy_info: Dictionary with energy information for labeling
    """
    os.makedirs(log_dir, exist_ok=True)

    status_dir = os.path.join(log_dir, status)
    os.makedirs(status_dir, exist_ok=True)

    fasta_path = os.path.join(status_dir, f"seq_iter_{iteration}_{status}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">Iteration_{iteration}_{status}\n{sequence}")

    final_pdb_path = os.path.join(status_dir, f"struct_iter_{iteration}_{status}.pdb")
    shutil.copy2(pdb_path, final_pdb_path)

    png_path = os.path.join(status_dir, f"image_iter_{iteration}_{status}.png")
    
    temp_dir = tempfile.mkdtemp()
    temp_png = os.path.join(temp_dir, f"temp_render_{status}.png")
    
    color_scheme = {
        "accepted": "chainbow",
        "rejected": "red", 
        "proposed": "yellow"
    }
    
    bg_color = "white"
    if status == "rejected":
        bg_color = "gray90"
    elif status == "proposed":
        bg_color = "lightyellow"
    
    title_text = f"Iter {iteration} - {status.upper()}"
    if energy_info:
        if 'total_energy' in energy_info:
            title_text += f" (E={energy_info['total_energy']:.2f})"
        if 'structure_score' in energy_info:
            title_text += f" (Score={energy_info['structure_score']:.1f})"

    script = f"""
load {final_pdb_path}
hide everything
show cartoon
set cartoon_smooth_loops, on
color {color_scheme.get(status, 'chainbow')}
bg_color {bg_color}
set ray_opaque_background, off

set label_size, 20
set label_color, black
pseudoatom title_label, pos=[0,0,30]
label title_label, "{title_text}"

orient
zoom all, 1.2
png {temp_png}, width=800, height=600, dpi=300, ray=1
quit
"""
        
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as script_file:
        script_file.write(script)
        script_path = script_file.name

    try:
        result = subprocess.run([
            PYMOL_BINARY, "-cq", script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if os.path.exists(temp_png) and os.path.getsize(temp_png) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
            shutil.copy2(temp_png, png_path)
            print(f"Rendered {status} image saved to {png_path}")
            return png_path
        else:
            print(f"PyMOL execution failed for {status} sequence.")
            return None
    except Exception as e:
        print(f"Error executing PyMOL for {status} sequence: {str(e)}")
        return None
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
        if os.path.exists(temp_png):
            os.remove(temp_png)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True) 