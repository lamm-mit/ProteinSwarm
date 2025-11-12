"""Distance matrix analysis for structural context."""

import numpy as np
from typing import Dict
from constants import DEFAULT_CUTOFF


def analyze_distance_matrix(dist_matrix: np.ndarray, sequence_length: int) -> Dict:
    """Analyze distance matrix to provide structural context for each position."""
    if dist_matrix is None or dist_matrix.size == 0:
        return {}
    
    analysis = {
        'compactness': {},
        'contact_density': {},
        'secondary_structure_region': {},
        'structural_flexibility': {}
    }
    contact_threshold = DEFAULT_CUTOFF
    
    for i in range(sequence_length):
        if i >= dist_matrix.shape[0]:
            continue
            
        local_distances = []
        for j in range(max(0, i-5), min(sequence_length, i+6)):
            if i != j and j < dist_matrix.shape[1]:
                local_distances.append(dist_matrix[i, j])
        
        if local_distances:
            avg_local_distance = np.mean(local_distances)
            if avg_local_distance < 6.0:
                compactness = "very_compact"
            elif avg_local_distance < 8.0:
                compactness = "compact"
            elif avg_local_distance < 12.0:
                compactness = "moderate"
            else:
                compactness = "extended"
        else:
            compactness = "unknown"
        
        analysis['compactness'][i] = compactness
        
        contacts = 0
        for j in range(sequence_length):
            if i != j and j < dist_matrix.shape[1] and dist_matrix[i, j] < contact_threshold:
                contacts += 1
        
        analysis['contact_density'][i] = contacts
        
        if i >= 3 and i < sequence_length - 3:
            consecutive_distances = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                j = i + offset
                if 0 <= j < sequence_length and j < dist_matrix.shape[1]:
                    consecutive_distances.append(dist_matrix[i, j])
            
            if consecutive_distances:
                avg_consecutive = np.mean(consecutive_distances)
                if avg_consecutive < 6.0:
                    ss_prediction = "helix-like"
                elif avg_consecutive > 12.0:
                    ss_prediction = "extended"
                else:
                    ss_prediction = "intermediate"
            else:
                ss_prediction = "unknown"
        else:
            ss_prediction = "terminal"
        
        analysis['secondary_structure_region'][i] = ss_prediction
        
        if local_distances and len(local_distances) > 2:
            distance_std = np.std(local_distances)
            if distance_std < 1.0:
                flexibility = "rigid"
            elif distance_std < 2.0:
                flexibility = "moderate"
            else:
                flexibility = "flexible"
        else:
            flexibility = "unknown"
        
        analysis['structural_flexibility'][i] = flexibility
    
    return analysis

def summarize_distance_matrix_analysis(analysis: Dict, position: int) -> str:
    """Create a human-readable summary of distance matrix analysis for a specific position."""
    if not analysis:
        return "No structural context available."
    
    summary_parts = []
    compactness = analysis.get('compactness', {}).get(position, 'unknown')
    contact_density = analysis.get('contact_density', {}).get(position, 0)
    ss_region = analysis.get('secondary_structure_region', {}).get(position, 'unknown')
    flexibility = analysis.get('structural_flexibility', {}).get(position, 'unknown')
    
    compactness_desc = {
        'very_compact': 'Very compact and tightly packed environment, average distance smaller than 6 Å',
        'compact': 'Compact structural environment, average distance 6-8 Å', 
        'moderate': 'Moderately packed environment, average distance 8-12 Å',
        'extended': 'Extended/loose structural environment, average distance larger than 12 Å',
        'unknown': 'Unknown packing environment'
    }
    summary_parts.append(f"- Compactness measured by average distance to 5 residues to the N-terminus and 5 residues to the C-terminus: {compactness_desc.get(compactness, compactness)}")
    
    if contact_density > 8:
        contact_desc = f"High contact density ({contact_density} neighbors) - highly constrained, more than 8 neighbors"
    elif contact_density > 4:
        contact_desc = f"Moderate contact density ({contact_density} neighbors) - moderately constrained, 4-8 neighbors"
    else:
        contact_desc = f"Low contact density ({contact_density} neighbors) - less constrained, less than 4 neighbors"
    summary_parts.append(f"- Contact density measured by number of neighbors within {DEFAULT_CUTOFF} Å: {contact_desc}")
    
    flex_desc = {
        'rigid': 'Structurally rigid position, standard deviation of local distances smaller than 1.0 Å ',
        'moderate': 'Moderately flexible position, standard deviation of local distances 1.0-2.0 Å',
        'flexible': 'Highly flexible position, standard deviation of local distances larger than 2.0 Å',
        'unknown': 'Unknown flexibility'
    }
    
    summary_parts.append(f"- Structural flexibility measured by standard deviation of local distances:: {flex_desc.get(flexibility, flexibility)}")
    
    ss_desc = {
        'helix-like': 'Local geometry potentially suggests helical environment',
        'extended': 'Local geometry potentially suggests extended or sheet-like environment',
        'intermediate': 'Potentially intermediate structural environment (turn/loop-like)',
        'terminal': 'Located at protein terminus',
        'unknown': 'Unknown local secondary structure'
    }
    summary_parts.append(f"- Secondary structure region predicted based on distance patterns: {ss_desc.get(ss_region, ss_region)}")
    
    return ".\n".join(summary_parts) + "."


def get_structural_recommendations(analysis: Dict, position: int, design_goal: str):
    """Get structural recommendations based on distance matrix analysis."""
    if not analysis:
        return []
    
    recommendations = []
    
    compactness = analysis.get('compactness', {}).get(position, 'unknown')
    contact_density = analysis.get('contact_density', {}).get(position, 0)
    ss_region = analysis.get('secondary_structure_region', {}).get(position, 'unknown')
    flexibility = analysis.get('structural_flexibility', {}).get(position, 'unknown')
    
    if compactness == 'very_compact':
        recommendations.append("Use small or hydrophobic residues due to tight packing")
    elif compactness == 'extended':
        recommendations.append("Larger residues acceptable in this extended environment")
    
    if contact_density > DEFAULT_CUTOFF:
        recommendations.append("Position is highly constrained - conservative mutations recommended")
    elif contact_density < DEFAULT_CUTOFF / 2:
        recommendations.append("Position has low constraints - more flexibility in amino acid choice")
    
    if "helix" in design_goal.lower() and ss_region == "helix-like":
        recommendations.append("Current geometry supports helix - maintain helix-compatible residues")
    elif "helix" in design_goal.lower() and ss_region == "extended":
        recommendations.append("Geometry not optimal for helix - consider helix-promoting residues")
    
    if "sheet" in design_goal.lower() and ss_region == "extended":
        recommendations.append("Current geometry supports sheet structure - maintain sheet-forming residues")
    
    if flexibility == 'rigid' and 'turn' in design_goal.lower():
        recommendations.append("Position is rigid - may need flexibility-promoting residues for turns")
    elif flexibility == 'flexible' and 'helix' in design_goal.lower():
        recommendations.append("Position is flexible - helix-promoting residues may help helix formation")
    elif flexibility == 'flexible' and 'sheet' in design_goal.lower():
        recommendations.append("Position is flexible - sheet-promoting residues may help sheet formation")
    
    return recommendations 