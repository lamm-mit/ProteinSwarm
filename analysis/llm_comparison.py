#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os

import umap
from sklearn.preprocessing import StandardScaler

MODEL_CONFIGS = {
    'grok-3-mini': {
        'csv_path': 'dummy_model1.csv',
        'short_name': 'grok-3-mini'
    },
    'GPT-o4-mini': {
        'csv_path': 'dummy_model2.csv',
        'short_name': 'GPT-o4-mini'
    },
    'Mistral-8B': {
        'csv_path': 'dummy_model3.csv',
        'short_name': 'Mistral-8B'
    },
    'GPT-4.1': {
        'csv_path': 'dummy_model4.csv',
        'short_name': 'GPT-4.1'
    },
    'GPT-4o': {
        'csv_path': 'dummy_model5.csv',
        'short_name': 'GPT-4o'
    },
    'Llama-3.2-3B': {
        'csv_path': 'dummy_model6.csv',
        'short_name': 'Llama-3.2-3B'
    },
}

def load_sequence_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    required_cols = ['iteration', 'sequence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df.dropna(subset=['sequence'])
    df = df.sort_values('iteration')
    
    return df

def hamming_distance(seq1: str, seq2: str) -> int:
    if len(seq1) != len(seq2):
        max_len = max(len(seq1), len(seq2))
        seq1 = seq1.ljust(max_len, 'X')
        seq2 = seq2.ljust(max_len, 'X')
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def calculate_hamming_distance_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    iteration_sequences = {}
    for _, row in df.iterrows():
        iteration = row['iteration']
        sequence = row['sequence']
        if iteration not in iteration_sequences:
            iteration_sequences[iteration] = sequence
    
    sorted_iterations = sorted(iteration_sequences.keys())
    sequences = [iteration_sequences[it] for it in sorted_iterations]
    
    n = len(sequences)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            distance = hamming_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = dist_matrix[j, i] = distance
    
    return dist_matrix, sorted_iterations

def physicochemical_encode_sequences(sequences: List[str]) -> np.ndarray:
    properties = {
        'A': [1.8, 0, 0, 0, 0],
        'R': [-4.5, 1, 1, 0, 0],
        'N': [-3.5, 0, 0, 1, 0],
        'D': [-3.5, -1, 0, 1, 0],
        'C': [2.5, 0, 0, 0, 1],
        'Q': [-3.5, 0, 0, 1, 0],
        'E': [-3.5, -1, 0, 1, 0],
        'G': [-0.4, 0, 0, 0, 0],
        'H': [-3.2, 1, 0.5, 0, 0],
        'I': [4.5, 0, 0, 0, 0],
        'L': [3.8, 0, 0, 0, 0],
        'K': [-3.9, 1, 1, 0, 0],
        'M': [1.9, 0, 0, 0, 1],
        'F': [2.8, 0, 0, 0, 0],
        'P': [-1.6, 0, 0, 0, 0],
        'S': [-0.8, 0, 0, 1, 0],
        'T': [-0.7, 0, 0, 1, 0],
        'W': [-0.9, 0, 0, 0, 0],
        'Y': [-1.3, 0, 0, 1, 0],
        'V': [4.2, 0, 0, 0, 0],
        'X': [0, 0, 0, 0, 0]
    }
    
    if not sequences:
        return np.array([])
    
    max_length = max(len(seq) for seq in sequences)
    n_properties = 5
    
    encoded = np.zeros((len(sequences), max_length * n_properties))
    
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j < max_length:
                props = properties.get(aa, properties['X'])
                start_idx = j * n_properties
                encoded[i, start_idx:start_idx + n_properties] = props
    
    return encoded

def compute_umap_embedding(sequences: List[str], n_neighbors: int = 15) -> Optional[np.ndarray]:
    if not sequences:
        return None
    
    X = physicochemical_encode_sequences(sequences)
    
    if X.size == 0:
        return None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples = X_scaled.shape[0]
    n_neighbors = min(n_neighbors, max(2, n_samples // 2))
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, 
                       min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    return embedding

def plot_hamming_distance_heatmap(ax, dist_matrix: np.ndarray, iterations: List[int], 
                                 title: str, vmin: float = None, vmax: float = None):
    im = ax.imshow(dist_matrix, cmap='coolwarm', aspect='auto', alpha=0.9, 
                   vmin=vmin, vmax=vmax)
    
    ax.set_title(title, fontsize=12, fontweight='bold', color='black')
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Iteration', fontsize=10)
    
    n_ticks = min(len(iterations), 8)
    tick_indices = np.linspace(0, len(iterations)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([str(iterations[i]) for i in tick_indices], fontsize=8)
    ax.set_yticklabels([str(iterations[i]) for i in tick_indices], fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Hamming Distance', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    return im

def plot_umap_clustering(ax, sequences: List[str], iterations: List[int], 
                        title: str):
    umap_embedding = compute_umap_embedding(sequences)
    
    if umap_embedding is not None:
        scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                           c=iterations, cmap='coolwarm', s=30, alpha=0.8, 
                           edgecolors='white', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Iteration', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', color='black')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'UMAP not available\nInstall umap-learn', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10, 
               style='italic')
        ax.set_title(f'{title} (UMAP Not Available)', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

def create_model_comparison_figure(output_file: str = "model_comparison_local_symmetry.png"):
    model_data = {}
    for model_name, config in MODEL_CONFIGS.items():
        csv_path = config['csv_path']
        if os.path.exists(csv_path):
            df = load_sequence_data(csv_path)
            if not df.empty:
                model_data[model_name] = {
                    'df': df,
                    'short_name': config['short_name']
                }
                print(f"Loaded {len(df)} records for {model_name}")
    
    if not model_data:
        print("Error: No valid model data found!")
        return
    
    all_dist_matrices = []
    
    for model_name in model_data.keys():
        data = model_data[model_name]
        df = data['df']
        dist_matrix, iterations = calculate_hamming_distance_matrix(df)
        all_dist_matrices.append(dist_matrix)
    
    all_dist_values = np.concatenate([matrix.flatten() for matrix in all_dist_matrices])
    hamming_vmin, hamming_vmax = np.min(all_dist_values), np.max(all_dist_values)
    
    print(f"Global Hamming distance scale: {hamming_vmin:.1f} to {hamming_vmax:.1f}")
    
    n_models = len(model_data)
    fig, axes = plt.subplots(n_models, 2, figsize=(7.5, 2.5*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, 2)
    
    model_names = list(model_data.keys())
    
    for i, model_name in enumerate(model_names):
        data = model_data[model_name]
        df = data['df']
        short_name = data['short_name']
        
        dist_matrix, iterations = calculate_hamming_distance_matrix(df)
        
        sequences = []
        seq_iterations = []
        iteration_sequences = {}
        for _, row in df.iterrows():
            iteration = row['iteration']
            sequence = row['sequence']
            if iteration not in iteration_sequences:
                iteration_sequences[iteration] = sequence
                
        for iteration in sorted(iteration_sequences.keys()):
            sequences.append(iteration_sequences[iteration])
            seq_iterations.append(iteration)
        
        plot_hamming_distance_heatmap(axes[i, 0], dist_matrix, iterations, 
                                     f'{short_name}',
                                     vmin=hamming_vmin, vmax=hamming_vmax)
        
        plot_umap_clustering(axes[i, 1], sequences, seq_iterations, 
                           f'{short_name}')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')
    print(f"Figure saved as: {output_file}")
    
    plt.show()
    
    return fig

def print_summary_statistics():
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY - LOCAL SYMMETRY DESIGN")
    print("="*80)
    
    for model_name, config in MODEL_CONFIGS.items():
        csv_path = config['csv_path']
        if os.path.exists(csv_path):
            df = load_sequence_data(csv_path)
            if not df.empty:
                print(f"\n📊 {model_name}:")
                print(f"  Total iterations: {df['iteration'].nunique()}")
                print(f"  Total sequences: {len(df)}")
                
                iteration_sequences = {}
                for _, row in df.iterrows():
                    iteration = row['iteration']
                    sequence = row['sequence']
                    if iteration not in iteration_sequences:
                        iteration_sequences[iteration] = sequence
                
                sorted_iterations = sorted(iteration_sequences.keys())
                consecutive_distances = []
                
                for i in range(len(sorted_iterations) - 1):
                    current_iter = sorted_iterations[i]
                    next_iter = sorted_iterations[i + 1]
                    seq_current = iteration_sequences[current_iter]
                    seq_next = iteration_sequences[next_iter]
                    distance = hamming_distance(seq_current, seq_next)
                    consecutive_distances.append(distance)
                
                if consecutive_distances:
                    print(f"  Mean consecutive Hamming distance: {np.mean(consecutive_distances):.2f}")
                    print(f"  Std consecutive Hamming distance: {np.std(consecutive_distances):.2f}")
                    print(f"  Min/Max consecutive distance: {min(consecutive_distances)}/{max(consecutive_distances)}")
            else:
                print(f"\n❌ {model_name}: No valid data")
        else:
            print(f"\n❌ {model_name}: File not found")
    
    print("="*80)

if __name__ == "__main__":
    print("Creating model comparison figure for local symmetry protein design...")
    
    print_summary_statistics()
    
    fig = create_model_comparison_figure()
    
    print("\n✅ Model comparison analysis completed!")
