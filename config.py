"""Configuration classes and data structures."""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
from pydantic import BaseModel, Field


def create_design_goal(name: str, description: str) -> Dict[str, str]:
    """Helper function to create a structured design goal."""
    return {
        'name': name,
        'description': description
    }


class DesignProposal(BaseModel):
    """Schema for structured output from LLM agent."""
    reasoning: str
    proposed_value: str = Field(..., description="A single one-letter amino acid code")


@dataclass
class DesignConfig:
    """Configuration for the design optimization loop."""
    system_description: str
    design_goal: Union[str, Dict[str, str]]
    update_strategy: str = "sequential"
    max_iterations: int = 3
    neighbor_radius: int = 1
    visualization: bool = True
    log_dir: str = "./design_log_nonlocal"
    pymol_binary: str = "pymol"
    current_dist_matrix: Optional[np.ndarray] = None
    current_pdb_path: Optional[str] = None
    residue_roles: Optional[Dict[int, str]] = None
    
    def get_goal_name(self) -> str:
        """Extract goal name from design_goal."""
        if isinstance(self.design_goal, dict):
            return self.design_goal.get('name', 'unknown')
        return str(self.design_goal)
    
    def get_goal_description(self) -> str:
        """Extract goal description from design_goal."""
        if isinstance(self.design_goal, dict):
            return self.design_goal.get('description', self.design_goal.get('name', 'unknown'))
        return str(self.design_goal)


class DesignGrid:
    """Grid representing the protein sequence and agent positions."""
    
    def __init__(self, sequence: str):
        self.sequence = list(sequence)
    
    def get_sequence(self) -> str:
        """Get current sequence as string."""
        return "".join(self.sequence)
    
    def positions(self):
        """Return all valid positions."""
        return range(len(self.sequence))
    
    def get_neighbors(self, index: int) -> Dict[str, str]:
        """Get linear sequence neighbors only."""
        from constants import DEFAULT_NEIGHBORS
        neighbors = {}
        neighbor_radius = int(DEFAULT_NEIGHBORS)
        
        n_term_neighbors = []
        for offset in range(-neighbor_radius, 0):
            neighbor_idx = index + offset
            if 0 <= neighbor_idx < len(self.sequence):
                n_term_neighbors.append(self.sequence[neighbor_idx])
        if n_term_neighbors:
            neighbors["n-term"] = "".join(n_term_neighbors)
                
        c_term_neighbors = []
        for offset in range(1, neighbor_radius + 1):
            neighbor_idx = index + offset
            if 0 <= neighbor_idx < len(self.sequence):
                c_term_neighbors.append(self.sequence[neighbor_idx])
        if c_term_neighbors:
            neighbors["c-term"] = "".join(c_term_neighbors)
                
        return neighbors
    
    def get_agent_input(self, index: int, config: DesignConfig, spatial_neighbors=None) -> Dict:
        """Get agent input for a specific position."""
        return {
            'position': index,
            'state': self.sequence[index],
            'sequence': self.get_sequence(),
            'neighbors': self.get_neighbors(index),
            'spatial_neighbors': spatial_neighbors,
            'goal': config.design_goal,
        }
    
    def update(self, index: int, new_state: str):
        """Update position with new amino acid."""
        if 0 <= index < len(self.sequence):
            self.sequence[index] = new_state
    
    def set_sequence(self, sequence: str):
        """Set the entire sequence."""
        self.sequence = list(sequence)
