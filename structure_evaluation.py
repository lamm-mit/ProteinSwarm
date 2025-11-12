"""Structure evaluation."""

import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from Bio.PDB import PDBParser, DSSP
from structure_utils import fallback_get_local_structure

def fix_pdb_format(input_pdb_file, output_pdb_file):
    """
    Fix PDB file format by adding required headers if missing.
    
    Args:
        input_pdb_file (str): Path to input PDB file
        output_pdb_file (str): Path to output fixed PDB file
        
    Returns:
        str: Path to the fixed PDB file
    """
    with open(input_pdb_file, 'r') as f:
        lines = f.readlines()
    
    has_header = any(line.startswith('HEADER') for line in lines[:10])
    has_cryst1 = any(line.startswith('CRYST1') for line in lines[:10])
    
    if has_header and has_cryst1:
        return input_pdb_file
    
    with open(output_pdb_file, 'w') as f:
        f.write("HEADER    PROTEIN STRUCTURE                        01-JAN-24   1ABC              \n")
        f.write("TITLE     DESIGNED PROTEIN STRUCTURE                                             \n")
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n")
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('ATOM') or line.startswith('TER') or line.startswith('END')):
                if line.startswith('ATOM') and len(line) < 80:
                    line = line.ljust(80)
                f.write(line + '\n')
        
        if not any(line.startswith('END') for line in lines[-5:]):
            f.write("END                                                                             \n")
    
    return output_pdb_file

def analyze_secondary_structure(pdb_path: str, dist_matrix: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze the secondary structure of a folded protein using DSSP.
    
    Args:
        pdb_path: Path to PDB file
        dist_matrix: Precomputed distance matrix (optional, kept for compatibility)
        
    Returns:
        Dictionary with secondary structure analysis
    """
    try:
        if not os.path.exists(pdb_path):
            return {
                'length': 0,
                'ss_string': '',
                'structure_assignment': [],
                'helix_count': 0,
                'sheet_count': 0,
                'loop_count': 0,
                'helix_regions': [],
                'sheet_regions': [],
                'turn_regions': [],
                'loop_regions': [],
                'num_helices': 0,
                'num_sheets': 0,
                'num_loops': 0,
                'helix_fraction': 0.0,
                'sheet_fraction': 0.0,
                'loop_fraction': 0.0,
                'error': f'PDB file not found: {pdb_path}'
            }
        
        if os.path.getsize(pdb_path) == 0:
            return {
                'length': 0,
                'ss_string': '',
                'structure_assignment': [],
                'helix_count': 0,
                'sheet_count': 0,
                'loop_count': 0,
                'helix_regions': [],
                'sheet_regions': [],
                'turn_regions': [],
                'loop_regions': [],
                'num_helices': 0,
                'num_sheets': 0,
                'num_loops': 0,
                'helix_fraction': 0.0,
                'sheet_fraction': 0.0,
                'loop_fraction': 0.0,
                'error': f'PDB file is empty: {pdb_path}'
            }
        
        fixed_pdb_file = pdb_path.replace('.pdb', '_fixed.pdb')
        try:
            actual_pdb_file = fix_pdb_format(pdb_path, fixed_pdb_file)
        except Exception as e:
            print(f"Warning: Could not fix PDB format: {e}")
            actual_pdb_file = pdb_path
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("PROTEIN", actual_pdb_file)
        model = structure[0]
        
        residues = list(structure.get_residues())
        length = len(residues)
        
        try:
            dssp = DSSP(model, actual_pdb_file)
        except Exception as e:
            print(f"Warning: DSSP analysis failed: {e}")
            if dist_matrix is not None:
                return _fallback_secondary_structure_analysis(pdb_path, dist_matrix, length)
            else:
                return _empty_secondary_structure_result(length)
        
        dssp_data = []
        for key in dssp.keys():
            ss = dssp[key][1]
            dssp_data.append({'secondary_structure': ss})
        
        df = pd.DataFrame(dssp_data)
        
        ss_mapping = {
            'H': 'helix',
            'G': 'helix',
            'I': 'helix',
            'B': 'sheet',
            'E': 'sheet',
            'T': 'turn',
            'S': 'turn',
            '-': 'turn'
        }
        
        df['ss_description'] = df['secondary_structure'].map(ss_mapping)
        df['ss_description'] = df['ss_description'].fillna('loop')
        
        structure_assignment = df['ss_description'].tolist()
        
        ss_string = ''.join(structure_assignment)
        
        helix_count = sum(1 for ss in structure_assignment if ss == 'helix')
        sheet_count = sum(1 for ss in structure_assignment if ss == 'sheet')
        turn_count = sum(1 for ss in structure_assignment if ss == 'turn')
        loop_count = sum(1 for ss in structure_assignment if ss == 'loop')
        
        helix_fraction = helix_count / length if length > 0 else 0.0
        sheet_fraction = sheet_count / length if length > 0 else 0.0
        turn_fraction = turn_count / length if length > 0 else 0.0
        loop_fraction = loop_count / length if length > 0 else 0.0
        
        helix_regions, sheet_regions, turn_regions, loop_regions = _find_secondary_structure_regions(structure_assignment)
        
        if actual_pdb_file != pdb_path and os.path.exists(actual_pdb_file):
            try:
                os.remove(actual_pdb_file)
            except:
                pass
        
        return {
            'length': length,
            'ss_string': ss_string,
            'structure_assignment': structure_assignment,
            'helix_count': helix_count,
            'sheet_count': sheet_count,
            'turn_count': turn_count,
            'loop_count': loop_count,
            'helix_fraction': helix_fraction,
            'sheet_fraction': sheet_fraction,
            'turn_fraction': turn_fraction,
            'loop_fraction': loop_fraction,
            'helix_regions': helix_regions,
            'sheet_regions': sheet_regions,
            'turn_regions': turn_regions,
            'loop_regions': loop_regions,
            'num_helices': len(helix_regions),
            'num_sheets': len(sheet_regions),
            'num_turns': len(turn_regions),
            'num_loops': len(loop_regions),
            'dssp_available': True
        }
        
    except Exception as e:
        print(f"Warning: Secondary structure analysis failed: {e}")
        if dist_matrix is not None:
            return _fallback_secondary_structure_analysis(pdb_path, dist_matrix, None)
        return _empty_secondary_structure_result(0)


def _find_secondary_structure_regions(structure_assignment: List[str]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Find continuous regions of secondary structures.
    
    Args:
        structure_assignment: List of secondary structure assignments
        
    Returns:
        Tuple of (helix_regions, sheet_regions, turn_regions, loop_regions)
    """
    helix_regions = []
    sheet_regions = []
    turn_regions = []
    loop_regions = []
    
    current_helix = []
    current_sheet = []
    current_turn = []
    current_loop = []
    
    for i, ss in enumerate(structure_assignment):
        if ss == 'helix':
            if current_sheet and len(current_sheet) >= 2:
                sheet_regions.append((current_sheet[0], current_sheet[-1]))
            if current_turn and len(current_turn) >= 1:
                turn_regions.append((current_turn[0], current_turn[-1]))
            if current_loop and len(current_loop) >= 1:
                loop_regions.append((current_loop[0], current_loop[-1]))
            current_sheet = []
            current_turn = []
            current_loop = []
            current_helix.append(i)
        elif ss == 'sheet':
            if current_helix and len(current_helix) >= 3:
                helix_regions.append((current_helix[0], current_helix[-1]))
            if current_turn and len(current_turn) >= 1:
                turn_regions.append((current_turn[0], current_turn[-1]))
            if current_loop and len(current_loop) >= 1:
                loop_regions.append((current_loop[0], current_loop[-1]))
            current_helix = []
            current_turn = []
            current_loop = []
            current_sheet.append(i)
        elif ss == 'turn':
            if current_helix and len(current_helix) >= 3:
                helix_regions.append((current_helix[0], current_helix[-1]))
            if current_sheet and len(current_sheet) >= 2:
                sheet_regions.append((current_sheet[0], current_sheet[-1]))
            if current_loop and len(current_loop) >= 1:
                loop_regions.append((current_loop[0], current_loop[-1]))
            current_helix = []
            current_sheet = []
            current_loop = []
            current_turn.append(i)
        else: 
            if current_helix and len(current_helix) >= 3:
                helix_regions.append((current_helix[0], current_helix[-1]))
            if current_sheet and len(current_sheet) >= 2:
                sheet_regions.append((current_sheet[0], current_sheet[-1]))
            if current_turn and len(current_turn) >= 1:
                turn_regions.append((current_turn[0], current_turn[-1]))
            current_helix = []
            current_sheet = []
            current_turn = []
            current_loop.append(i)

    if len(current_helix) >= 3:
        helix_regions.append((current_helix[0], current_helix[-1]))
    if len(current_sheet) >= 2:
        sheet_regions.append((current_sheet[0], current_sheet[-1]))
    if len(current_turn) >= 1:
        turn_regions.append((current_turn[0], current_turn[-1]))
    if len(current_loop) >= 1:
        loop_regions.append((current_loop[0], current_loop[-1]))
    
    return helix_regions, sheet_regions, turn_regions, loop_regions


def _fallback_secondary_structure_analysis(pdb_path: str, dist_matrix: np.ndarray, length: Optional[int] = None) -> Dict:
    """
    Fallback secondary structure analysis using distance matrix.
    
    Args:
        pdb_path: Path to PDB file
        dist_matrix: Distance matrix
        length: Protein length (if known)
        
    Returns:
        Dictionary with secondary structure analysis
    """
    try:
        if length is None:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("model", pdb_path)
            residues = list(structure.get_residues())
            length = len(residues)
        
        structure_assignment = []
        for i in range(length):
            ss = fallback_get_local_structure(i, dist_matrix)
            structure_assignment.append(ss)
        
        ss_string = ''.join(structure_assignment)
        
        helix_count = ss_string.count('helix')
        sheet_count = ss_string.count('sheet') 
        loop_count = ss_string.count('loop')
        
        helix_fraction = helix_count / length if length > 0 else 0.0
        sheet_fraction = sheet_count / length if length > 0 else 0.0
        loop_fraction = loop_count / length if length > 0 else 0.0
        
        helix_regions, sheet_regions, turn_regions, loop_regions = _find_secondary_structure_regions(structure_assignment)
        
        return {
            'length': length,
            'ss_string': ss_string,
            'structure_assignment': structure_assignment,
            'helix_count': helix_count,
            'sheet_count': sheet_count,
            'loop_count': loop_count,
            'helix_fraction': helix_fraction,
            'sheet_fraction': sheet_fraction,
            'loop_fraction': loop_fraction,
            'helix_regions': helix_regions,
            'sheet_regions': sheet_regions,
            'turn_regions': turn_regions,
            'loop_regions': loop_regions,
            'num_helices': len(helix_regions),
            'num_sheets': len(sheet_regions),
            'num_turns': len(turn_regions),
            'num_loops': len(loop_regions),
            'dssp_available': False
        }
        
    except Exception as e:
        print(f"Warning: Fallback secondary structure analysis failed: {e}")
        return _empty_secondary_structure_result(length or 0)


def _empty_secondary_structure_result(length: int) -> Dict:
    """
    Return empty secondary structure result.
    
    Args:
        length: Protein length
        
    Returns:
        Dictionary with empty secondary structure data
    """
    return {
        'length': length,
        'ss_string': '',
        'structure_assignment': ['unknown'] * length,
        'helix_count': 0,
        'sheet_count': 0,
        'turn_count': 0,
        'loop_count': 0,
        'helix_fraction': 0.0,
        'sheet_fraction': 0.0,
        'turn_fraction': 0.0,
        'loop_fraction': 0.0,
        'helix_regions': [],
        'sheet_regions': [],
        'turn_regions': [],
        'loop_regions': [],
        'num_helices': 0,
        'num_sheets': 0,
        'num_turns': 0,
        'num_loops': 0,
        'dssp_available': False
    }


def analyze_sequence_composition(sequence: str) -> Dict:
    """
    Analyze amino acid composition and patterns in the sequence.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Dictionary with sequence composition analysis
    """
    hydrophobic = set('AILGMFPWVY')
    polar = set('STNQRHKDEC')
    charged = set('RHKDE')
    aromatic = set('FWY')
    small = set('AGCS')
    flexible = set('GP')
    helix_forming = set('ALMEK')
    sheet_forming = set('IVYFWT')
    turn_forming = set('GP')
    
    length = len(sequence)
    composition = {}
    
    composition['hydrophobic_count'] = sum(1 for aa in sequence if aa in hydrophobic)
    composition['polar_count'] = sum(1 for aa in sequence if aa in polar)
    composition['charged_count'] = sum(1 for aa in sequence if aa in charged)
    composition['aromatic_count'] = sum(1 for aa in sequence if aa in aromatic)
    composition['small_count'] = sum(1 for aa in sequence if aa in small)
    composition['flexible_count'] = sum(1 for aa in sequence if aa in flexible)
    composition['helix_forming_count'] = sum(1 for aa in sequence if aa in helix_forming)
    composition['sheet_forming_count'] = sum(1 for aa in sequence if aa in sheet_forming)
    composition['turn_forming_count'] = sum(1 for aa in sequence if aa in turn_forming)
    
    composition['hydrophobic_fraction'] = composition['hydrophobic_count'] / length
    composition['polar_fraction'] = composition['polar_count'] / length
    composition['charged_fraction'] = composition['charged_count'] / length
    composition['aromatic_fraction'] = composition['aromatic_count'] / length
    composition['small_fraction'] = composition['small_count'] / length
    composition['flexible_fraction'] = composition['flexible_count'] / length
    composition['helix_forming_fraction'] = composition['helix_forming_count'] / length
    composition['sheet_forming_fraction'] = composition['sheet_forming_count'] / length
    composition['turn_forming_fraction'] = composition['turn_forming_count'] / length
    
    composition['unique_residues'] = len(set(sequence))
    composition['diversity_fraction'] = composition['unique_residues'] / 20
    
    composition['alternating_patterns'] = []
    for i in range(length - 1):
        aa1, aa2 = sequence[i], sequence[i+1]
        if aa1 in hydrophobic and aa2 in polar:
            composition['alternating_patterns'].append(('hydrophobic-polar', i))
        elif aa1 in polar and aa2 in hydrophobic:
            composition['alternating_patterns'].append(('polar-hydrophobic', i))
    
    composition['repeating_patterns'] = []
    for pattern_len in range(2, max(3, len(sequence) // 3 + 1)):
        for i in range(length - pattern_len * 2 + 1):
            pattern = sequence[i:i+pattern_len]
            if sequence[i+pattern_len:i+pattern_len*2] == pattern:
                composition['repeating_patterns'].append((pattern, i))

    composition['repetitive_residues'] = []
    for aa in set(sequence):
        count = sequence.count(aa)
        if count > length * 0.3:
            composition['repetitive_residues'].append((aa, count, count/length))
    
    return composition


def evaluate_design_goal(design_goal: Union[str, Dict[str, str]], pdb_path: str, sequence: str, dist_matrix: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate how well a structure matches a general design goal.
    
    Args:
        design_goal: The design goal string or dictionary with 'name' and 'description'
        pdb_path: Path to PDB file
        sequence: Protein sequence
        dist_matrix: Precomputed distance matrix (optional)
        
    Returns:
        Dictionary with evaluation results
    """
    if isinstance(design_goal, dict):
        design_goal_str = design_goal.get('description', design_goal.get('name', str(design_goal)))
    else:
        design_goal_str = str(design_goal)
    
    ss_analysis = analyze_secondary_structure(pdb_path, dist_matrix)
    seq_analysis = analyze_sequence_composition(sequence)
    
    dist_matrix_analysis = {}
    if dist_matrix is not None and dist_matrix.size > 0:
        from distance_matrix_analysis import analyze_distance_matrix, summarize_distance_matrix_analysis, get_structural_recommendations
        dist_matrix_analysis = analyze_distance_matrix(dist_matrix, len(sequence))
    
    evaluation = {
        'design_goal': design_goal_str,
        'sequence': sequence,
        'structure_analysis': ss_analysis,
        'secondary_structure_analysis': ss_analysis,
        'sequence_analysis': seq_analysis,
        'distance_matrix_analysis': dist_matrix_analysis,
        'goal_match': False,
        'score': 0.0,
        'details': {},
        'recommendations': [],
        'position_specific_context': {}
    }
    
    try:
        score = 0.0
        details = {}
        recommendations = []
        max_possible_score = 0.0
        triggered_conditions = []
        
        design_goal_lower = design_goal_str.lower()
        
        if any(keyword in design_goal_lower for keyword in ['compact', 'packing', 'hydrophobic']):
            hydrophobic_fraction = seq_analysis['hydrophobic_fraction']
            max_possible_score += 25.0
            triggered_conditions.append('hydrophobic')

            if hydrophobic_fraction >= 0.6:
                score += 25.0
                details['hydrophobic_content'] = f"Good hydrophobic content: {hydrophobic_fraction:.1%}"
            elif hydrophobic_fraction >= 0.5:
                score += 20.0
                details['hydrophobic_content'] = f"Somewhat good hydrophobic content: {hydrophobic_fraction:.1%}"
                recommendations.append("Increase hydrophobic residues for better compact packing")
            elif hydrophobic_fraction >= 0.4:
                score += 15.0
                details['hydrophobic_content'] = f"Moderate hydrophobic content: {hydrophobic_fraction:.1%}"
                recommendations.append("Increase hydrophobic residues for better compact packing")
            elif hydrophobic_fraction >= 0.3:
                score += 10.0
                details['hydrophobic_content'] = f"Somewhat low hydrophobic content: {hydrophobic_fraction:.1%}"
                recommendations.append("Increase hydrophobic residues for better compact packing")
            else:
                details['hydrophobic_content'] = f"Low hydrophobic content: {hydrophobic_fraction:.1%}"
                recommendations.append("Increase hydrophobic residues for better compact packing")
        
        if any(keyword in design_goal_lower for keyword in ['beta', 'strand', 'sheet']):
            sheet_fraction = ss_analysis['sheet_count'] / max(ss_analysis['length'], 1)
            sheet_forming_fraction = seq_analysis['sheet_forming_fraction']
            max_possible_score += 25.0
            triggered_conditions.append('beta_sheets')
            
            if sheet_fraction >= 0.6 and sheet_forming_fraction >= 0.6:
                score += 25.0
                details['beta_structure'] = f"Good beta content: {sheet_fraction:.1%} structure, {sheet_forming_fraction:.1%} sheet-forming residues"
            elif sheet_fraction >= 0.4 and sheet_forming_fraction >= 0.4:
                score += 20.0
                details['beta_structure'] = f"Somewhat good beta content: {sheet_fraction:.1%} structure, {sheet_forming_fraction:.1%} sheet-forming residues"
                recommendations.append("Increase sheet-forming residues (I,V,Y,F,W,T) for better beta structure")
            elif sheet_fraction >= 0.2 and sheet_forming_fraction >= 0.2:
                score += 15.0
                details['beta_structure'] = f"Some beta content: {sheet_fraction:.1%} structure, {sheet_forming_fraction:.1%} sheet-forming residues"
                recommendations.append("Increase sheet-forming residues (I,V,Y,F,W,T) for better beta structure")
            elif sheet_fraction >= 0.1 and sheet_forming_fraction >= 0.1:
                score += 10.0
                details['beta_structure'] = f"Somewhat limited beta content: {sheet_fraction:.1%} structure, {sheet_forming_fraction:.1%} sheet-forming residues"
                recommendations.append("Increase sheet-forming residues (I,V,Y,F,W,T) for better beta structure")
            else:
                details['beta_structure'] = f"Limited beta content: {sheet_fraction:.1%} structure, {sheet_forming_fraction:.1%} sheet-forming residues"
                recommendations.append("Increase sheet-forming residues (I,V,Y,F,W,T) for better beta structure")
        
        if any(keyword in design_goal_lower for keyword in ['alpha', 'helix', 'helical']):
            helix_fraction = ss_analysis['helix_count'] / max(ss_analysis['length'], 1)
            helix_forming_fraction = seq_analysis['helix_forming_fraction']
            max_possible_score += 25.0
            triggered_conditions.append('alpha_helices')
            
            if helix_fraction >= 0.6 and helix_forming_fraction >= 0.6:
                score += 25.0
                details['helix_structure'] = f"Good helix content: {helix_fraction:.1%} structure, {helix_forming_fraction:.1%} helix-forming residues"
            elif helix_fraction >= 0.4 and helix_forming_fraction >= 0.4:
                score += 20.0
                details['helix_structure'] = f"Somewhat good helix content: {helix_fraction:.1%} structure, {helix_forming_fraction:.1%} helix-forming residues"
                recommendations.append("Increase helix-forming residues (A,L,E,M,K) for better alpha helix structure")
            elif helix_fraction >= 0.2 and helix_forming_fraction >= 0.2:
                score += 15.0
                details['helix_structure'] = f"Some helix content: {helix_fraction:.1%} structure, {helix_forming_fraction:.1%} helix-forming residues"
                recommendations.append("Increase helix-forming residues (A,L,E,M,K) for better alpha helix structure")
            elif helix_fraction >= 0.1 and helix_forming_fraction >= 0.1:
                score += 10.0
                details['helix_structure'] = f"Somewhat limited helix content: {helix_fraction:.1%} structure, {helix_forming_fraction:.1%} helix-forming residues"
                recommendations.append("Increase helix-forming residues (A,L,E,M,K) for better alpha helix structure")
            else:
                details['helix_structure'] = f"Limited helix content: {helix_fraction:.1%} structure, {helix_forming_fraction:.1%} helix-forming residues"
                recommendations.append("Increase helix-forming residues (A,L,E,M,K) for better alpha helix structure")
        
        if any(keyword in design_goal_lower for keyword in ['flexible', 'hinge', 'turn']):
            turn_fraction = ss_analysis.get('turn_count', 0) / max(ss_analysis['length'], 1)
            turn_regions_count = ss_analysis.get('num_turns', 0)
            max_possible_score += 25.0
            triggered_conditions.append('flexibility')
            
            if  turn_fraction >= 0.7:
                score += 25.0
                details['flexibility'] = f"Good turn content: {turn_fraction:.1%} turns ({turn_regions_count} regions)"
            elif turn_fraction >= 0.5:
                score += 20.0
                details['flexibility'] = f"Somewhat good turn content: {turn_fraction:.1%} turns ({turn_regions_count} regions)"
                recommendations.append("Increase turns/hinges (with G,P) for more flexibility")
            elif turn_fraction >= 0.3:
                score += 15.0
                details['flexibility'] = f"Some turn content: {turn_fraction:.1%} turns ({turn_regions_count} regions)"
                recommendations.append("Increase turns/hinges (with G,P) for more flexibility")
            elif turn_fraction >= 0.1:
                score += 10.0
                details['flexibility'] = f"Limited turn content: {turn_fraction:.1%} turns ({turn_regions_count} regions)"
                recommendations.append("Increase turns/hinges (with G,P) for more flexibility")
            else:
                details['flexibility'] = f"Very limited turn content: {turn_fraction:.1%} turns ({turn_regions_count} regions)"
                recommendations.append("Increase turns/hinges (with G,P) for more flexibility")
        
        if any(keyword in design_goal_lower for keyword in ['alternating', 'alternate']):
            alternating_count = len(seq_analysis['alternating_patterns'])
            max_possible_score += 25.0
            triggered_conditions.append('alternating')
            
            if alternating_count >= 4:
                score += 25.0
                details['alternating_pattern'] = f"Good alternating pattern: {alternating_count} alternations"
            elif alternating_count >= 3:
                score += 15.0
                details['alternating_pattern'] = f"Somewhat good alternating pattern: {alternating_count} alternations"
                recommendations.append("Increase alternating hydrophobic-polar patterns")
            elif alternating_count >= 2:
                score += 10.0
                details['alternating_pattern'] = f"Some alternating pattern: {alternating_count} alternations"
                recommendations.append("Increase alternating hydrophobic-polar patterns")
            else:
                details['alternating_pattern'] = f"Limited alternating pattern: {alternating_count} alternations"
                recommendations.append("Increase alternating hydrophobic-polar patterns")
        
        if any(keyword in design_goal_lower for keyword in ['repeat', 'pattern']):
            repeating_count = len(seq_analysis['repeating_patterns'])
            max_possible_score += 25.0
            triggered_conditions.append('repeating')
            
            if repeating_count >= 4:
                score += 25.0
                details['repeating_patterns'] = f"Good repeating pattern: Found {repeating_count} repeating patterns"
            elif repeating_count >= 3:
                score += 20.0
                details['repeating_patterns'] = f"Somewhat good repeating pattern: Found {repeating_count} repeating patterns"
                recommendations.append("Introduce more repeating sequence motifs")
            elif repeating_count >= 2:
                score += 15.0
                details['repeating_patterns'] = f"Some repeating pattern: Found {repeating_count} repeating patterns"
                recommendations.append("Introduce more repeating sequence motifs")
            elif repeating_count >= 1:
                score += 10.0
                details['repeating_patterns'] = f"Somewhat limited repeating pattern: Found {repeating_count} repeating patterns"
                recommendations.append("Introduce more repeating sequence motifs")
            else:
                details['repeating_patterns'] = f"No clear repeating patterns found"
                recommendations.append("Introduce more repeating sequence motifs")
        
        if any(keyword in design_goal_lower for keyword in ['structure', 'rigid']):
            structured_fraction = (ss_analysis['helix_count'] + ss_analysis['sheet_count']) / max(ss_analysis['length'], 1)
            max_possible_score += 25.0
            triggered_conditions.append('structure_promotion')
            
            if structured_fraction >= 0.6:
                score += 25.0
                details['structure_promotion'] = f"Well-structured: {structured_fraction:.1%} structured"
            elif structured_fraction >= 0.4:
                score += 20.0
                details['structure_promotion'] = f"Moderately structured: {structured_fraction:.1%} structured"
                recommendations.append("Increase structured regions with either helices (A,L,E,M,K) or sheets (I,V,Y,F,W,T)")
            elif structured_fraction >= 0.3:
                score += 15.0
                details['structure_promotion'] = f"Somewhat structured: {structured_fraction:.1%} structured"
                recommendations.append("Increase structured regions with either helices (A,L,E,M,K) or sheets (I,V,Y,F,W,T)")
            elif structured_fraction >= 0.2:
                score += 10.0
                details['structure_promotion'] = f"Not so structured: {structured_fraction:.1%} structured"
                recommendations.append("Increase structured regions with either helices (A,L,E,M,K) or sheets (I,V,Y,F,W,T)")
            else:
                details['structure_promotion'] = f"Very low structured regions: {structured_fraction:.1%} structured"
                recommendations.append("Increase structured regions and amino acid diversity")
        
        if any(keyword in design_goal_lower for keyword in ['charged', 'polar', 'ionic']):
            charged_polar_fraction = seq_analysis['charged_fraction'] + seq_analysis['polar_fraction']
            max_possible_score += 25.0
            triggered_conditions.append('charged_polar')
            
            if charged_polar_fraction >= 0.6:
                score += 25.0
                details['charged_residues'] = f"Good charged content: {charged_polar_fraction:.1%} charged and polar"
            elif charged_polar_fraction >= 0.4:
                score += 20.0
                details['charged_residues'] = f"Somewhat good charged content: {charged_polar_fraction:.1%} charged and polar"
                recommendations.append("Incorporate more charged and polar residues (S,T,N,Q,R,H,K,D,E,C)")
            elif charged_polar_fraction >= 0.3:
                score += 15.0
                details['charged_residues'] = f"Some charged content: {charged_polar_fraction:.1%} charged and polar"
                recommendations.append("Incorporate more charged and polar residues (S,T,N,Q,R,H,K,D,E,C)")
            elif charged_polar_fraction >= 0.2:
                score += 10.0
                details['charged_residues'] = f"Somewhat low charged content: {charged_polar_fraction:.1%} charged and polar"
                recommendations.append("Incorporate more charged and polar residues (S,T,N,Q,R,H,K,D,E,C)")
            else:
                details['charged_residues'] = f"Low charged content: {charged_polar_fraction:.1%} charged and polar"
                recommendations.append("Increase charged residues for more extended structure")
        
        if any(keyword in design_goal_lower for keyword in ['symmetry', 'mirror', 'neighbor', 'palindromic']):
            symmetric_score = 0
            for i in range(1, len(sequence) - 1):
                if sequence[i-1] == sequence[i+1]:
                    symmetric_score += 1
            
            symmetry_fraction = symmetric_score / max(len(sequence) - 2, 1)
            max_possible_score += 25.0
            triggered_conditions.append('symmetry')

            if symmetry_fraction >= 0.8:
                score += 25.0
                details['local_symmetry'] = f"Good local symmetry: {symmetry_fraction:.1%}"
            elif symmetry_fraction >= 0.6:
                score += 20.0
                details['local_symmetry'] = f"Somewhat good local symmetry: {symmetry_fraction:.1%}"
                recommendations.append("Design sequences where residues mirror their neighbors")
            elif symmetry_fraction >= 0.4:
                score += 15.0
                details['local_symmetry'] = f"Some local symmetry: {symmetry_fraction:.1%}"
                recommendations.append("Design sequences where residues mirror their neighbors")
            elif symmetry_fraction >= 0.2:
                score += 10.0
                details['local_symmetry'] = f"Somewhat limited local symmetry: {symmetry_fraction:.1%}"
                recommendations.append("Design sequences where residues mirror their neighbors")
            else:
                details['local_symmetry'] = f"Limited local symmetry: {symmetry_fraction:.1%}"
                recommendations.append("Design sequences where residues mirror their neighbors")

        if any(keyword in design_goal_lower for keyword in ['beta hairpin', 'hairpin', 'beta_hairpin']):
            hairpin_score = 0
            aromatic_in_strands = 0
            proline_in_turns = 0
            
            aromatic_residues = set(['F', 'Y', 'W'])
            for aa in sequence:
                if aa in aromatic_residues:
                    aromatic_in_strands += 1
            
            proline_in_turns = sequence.count('P')
            
            if ss_analysis and 'sheet_regions' in ss_analysis:
                sheet_regions = ss_analysis['sheet_regions']
                if len(sheet_regions) >= 2:
                    hairpin_score += 15
                elif len(sheet_regions) == 1:
                    hairpin_score += 10
            
            max_possible_score += 30.0
            triggered_conditions.append('beta_hairpin')
            
            if aromatic_in_strands == 2 and proline_in_turns == 1:
                hairpin_score += 15
                details['beta_hairpin'] = f"Good hairpin features: {aromatic_in_strands} aromatic residues, {proline_in_turns} proline(s)"
            elif aromatic_in_strands == 1 and proline_in_turns == 1:
                hairpin_score += 10
                details['beta_hairpin'] = f"Some hairpin features: {aromatic_in_strands} aromatic residues, {proline_in_turns} proline(s)"
                recommendations.append("Add more aromatic residues (F,Y,W) in strands and proline (P) in turns")
            else:
                details['beta_hairpin'] = f"Limited hairpin features: {aromatic_in_strands} aromatic residues, {proline_in_turns} proline(s)"
                recommendations.append("Add aromatic residues (F,Y,W) for strand stability and proline (P) for turns")
            
            score += hairpin_score

        if any(keyword in design_goal_lower for keyword in ['helix-turn-helix', 'hth', 'helix turn helix']):
            hth_score = 0
            
            helix_residues = set(['A', 'L', 'E', 'M', 'K', 'Q', 'R'])
            turn_residues = set(['G', 'P'])
            
            seq_len = len(sequence)
            first_third = sequence[:seq_len//3]
            middle_third = sequence[seq_len//3:2*seq_len//3]
            last_third = sequence[2*seq_len//3:]
            
            helix1_fraction = sum(1 for aa in first_third if aa in helix_residues) / len(first_third)
            turn_fraction = sum(1 for aa in middle_third if aa in turn_residues) / len(middle_third)
            helix2_fraction = sum(1 for aa in last_third if aa in helix_residues) / len(last_third)
            
            max_possible_score += 30.0
            triggered_conditions.append('hth_motif')
            
            if helix1_fraction >= 0.6 and helix2_fraction >= 0.6 and turn_fraction >= 0.4:
                hth_score = 30
                details['hth_motif'] = f"Excellent HTH pattern: H1({helix1_fraction:.1%}) T({turn_fraction:.1%}) H2({helix2_fraction:.1%})"
            elif helix1_fraction >= 0.4 and helix2_fraction >= 0.4:
                hth_score = 20
                details['hth_motif'] = f"Good HTH pattern: H1({helix1_fraction:.1%}) T({turn_fraction:.1%}) H2({helix2_fraction:.1%})"
                recommendations.append("Enhance turn region with more flexible residues (G,P,S,T,N,D)")
            else:
                hth_score = 10
                details['hth_motif'] = f"Developing HTH pattern: H1({helix1_fraction:.1%}) T({turn_fraction:.1%}) H2({helix2_fraction:.1%})"
                recommendations.append("Improve helix regions with A,L,E,M,K,Q,R and turn with G,P,S,T,N,D")
            
            score += hth_score

        if any(keyword in design_goal_lower for keyword in ['compact hydrophobic', 'hydrophobic loop', 'compact_hydrophobic']):
            hydrophobic_fraction = seq_analysis['hydrophobic_fraction']
            loop_residues = set(['G', 'P', 'S', 'T', 'N', 'D'])
            loop_fraction = sum(1 for aa in sequence if aa in loop_residues) / len(sequence)
            
            max_possible_score += 25.0
            triggered_conditions.append('compact_hydrophobic')
            
            if hydrophobic_fraction >= 0.5 and loop_fraction >= 0.3:
                score += 25
                details['compact_hydrophobic'] = f"Good compact hydrophobic loop: {hydrophobic_fraction:.1%} hydrophobic, {loop_fraction:.1%} loop-promoting"
            elif hydrophobic_fraction >= 0.4:
                score += 20
                details['compact_hydrophobic'] = f"Moderate hydrophobic content: {hydrophobic_fraction:.1%} hydrophobic, {loop_fraction:.1%} loop-promoting"
                recommendations.append("Increase loop-promoting residues (G,P,S,T,N,D,Q) for better loop formation")
            else:
                score += 10
                details['compact_hydrophobic'] = f"Limited hydrophobic loop features: {hydrophobic_fraction:.1%} hydrophobic, {loop_fraction:.1%} loop-promoting"
                recommendations.append("Increase hydrophobic residues (I,V,L,F,M,W,Y,A) and loop residues (G,P,S,T,N,D,Q)")

        if any(keyword in design_goal_lower for keyword in ['loose coil', 'extended coil', 'loose_coil']):
            charged_polar = seq_analysis['charged_fraction'] + seq_analysis['polar_fraction']
            flexible_fraction = seq_analysis['flexible_fraction']
            
            compact_residues = set(['I', 'V', 'L', 'F', 'M', 'W'])
            compact_fraction = sum(1 for aa in sequence if aa in compact_residues) / len(sequence)
            
            max_possible_score += 25.0
            triggered_conditions.append('loose_coils')
            
            if charged_polar >= 0.6 and flexible_fraction >= 0.4 and compact_fraction <= 0.3:
                score += 25
                details['loose_coils'] = f"Excellent loose coil features: {charged_polar:.1%} charged/polar, {flexible_fraction:.1%} flexible"
            elif charged_polar >= 0.4 and flexible_fraction >= 0.3:
                score += 20
                details['loose_coils'] = f"Good loose coil features: {charged_polar:.1%} charged/polar, {flexible_fraction:.1%} flexible"
                recommendations.append("Reduce compact residues (I,V,L,F,M,W) for more extended structure")
            else:
                score += 10
                details['loose_coils'] = f"Limited loose coil features: {charged_polar:.1%} charged/polar, {flexible_fraction:.1%} flexible"
                recommendations.append("Increase charged/polar residues (R,K,D,E,S,T,N,Q) and reduce compact residues")

        if any(keyword in design_goal_lower for keyword in ['alternating charge', 'charge alternation', 'alternating_charge']):
            positive_charges = set(['R', 'K', 'H'])
            negative_charges = set(['D', 'E'])
            
            alternating_score = 0
            charge_pairs = 0
            
            for i in range(len(sequence) - 1):
                curr_aa = sequence[i]
                next_aa = sequence[i + 1]
                
                if ((curr_aa in positive_charges and next_aa in negative_charges) or 
                    (curr_aa in negative_charges and next_aa in positive_charges)):
                    alternating_score += 1
                    charge_pairs += 1
            
            alternation_fraction = alternating_score / max(len(sequence) - 1, 1)
            total_charged = sum(1 for aa in sequence if aa in positive_charges or aa in negative_charges)
            
            max_possible_score += 25.0
            triggered_conditions.append('alternating_charge')
            
            if alternation_fraction >= 0.4 and total_charged >= len(sequence) * 0.6:
                score += 25
                details['alternating_charge'] = f"Excellent charge alternation: {alternation_fraction:.1%} alternating, {charge_pairs} pairs"
            elif alternation_fraction >= 0.2:
                score += 20
                details['alternating_charge'] = f"Good charge alternation: {alternation_fraction:.1%} alternating, {charge_pairs} pairs"
                recommendations.append("Increase alternating positive (R,K,H) and negative (D,E) charges")
            else:
                score += 10
                details['alternating_charge'] = f"Limited charge alternation: {alternation_fraction:.1%} alternating, {charge_pairs} pairs"
                recommendations.append("Create alternating pattern of positive (R,K,H) and negative (D,E) charges")

        if any(keyword in design_goal_lower for keyword in ['helix cap', 'cap stabilization', 'helix_cap']):
            n_cap_residues = set(['S', 'T', 'N', 'D'])
            c_cap_residues = set(['G', 'P', 'S', 'T', 'N', 'D'])
            helix_residues = set(['A', 'L', 'E', 'M', 'K', 'Q', 'R'])
            
            cap_score = 0
            seq_len = len(sequence)
            
            n_term_caps = sum(1 for aa in sequence[:3] if aa in n_cap_residues)
            
            c_term_caps = sum(1 for aa in sequence[-3:] if aa in c_cap_residues)
            
            middle_helix = sum(1 for aa in sequence[3:-3] if aa in helix_residues) / max(len(sequence[3:-3]), 1)
            
            max_possible_score += 25.0
            triggered_conditions.append('helix_cap')
            
            if n_term_caps >= 1 and c_term_caps >= 1 and middle_helix >= 0.5:
                cap_score = 25
                details['helix_cap'] = f"Excellent helix capping: N-cap({n_term_caps}), C-cap({c_term_caps}), helix content({middle_helix:.1%})"
            elif (n_term_caps >= 1 or c_term_caps >= 1) and middle_helix >= 0.3:
                cap_score = 20
                details['helix_cap'] = f"Good helix capping: N-cap({n_term_caps}), C-cap({c_term_caps}), helix content({middle_helix:.1%})"
                recommendations.append("Add more N-cap (S,T,N,D) and C-cap (G,P,S,T,N,D) residues")
            else:
                cap_score = 10
                details['helix_cap'] = f"Limited helix capping: N-cap({n_term_caps}), C-cap({c_term_caps}), helix content({middle_helix:.1%})"
                recommendations.append("Add N-cap residues (S,T,N,D) at start and C-cap residues (G,P,S,T,N,D) at end")
            
            score += cap_score

        if any(keyword in design_goal_lower for keyword in ['amphipathic', 'amphipathic helix']):
            hydrophobic_residues = set(['I', 'V', 'L', 'F', 'M', 'W', 'Y', 'A'])
            polar_residues = set(['S', 'T', 'N', 'Q', 'R', 'H', 'K', 'D', 'E', 'C'])
            
            hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_residues)
            polar_count = sum(1 for aa in sequence if aa in polar_residues)
            
            hydrophobic_fraction = hydrophobic_count / len(sequence)
            polar_fraction = polar_count / len(sequence)
            
            max_possible_score += 25.0
            triggered_conditions.append('amphipathic_helix')
            
            if 0.4 <= hydrophobic_fraction <= 0.6 and 0.4 <= polar_fraction <= 0.6:
                score += 25
                details['amphipathic_helix'] = f"Excellent amphipathic balance: {hydrophobic_fraction:.1%} hydrophobic, {polar_fraction:.1%} polar"
            elif hydrophobic_fraction >= 0.3 and polar_fraction >= 0.3:
                score += 20
                details['amphipathic_helix'] = f"Good amphipathic balance: {hydrophobic_fraction:.1%} hydrophobic, {polar_fraction:.1%} polar"
                recommendations.append("Balance hydrophobic and polar residues for amphipathic character")
            else:
                score += 10
                details['amphipathic_helix'] = f"Limited amphipathic character: {hydrophobic_fraction:.1%} hydrophobic, {polar_fraction:.1%} polar"
                recommendations.append("Add both hydrophobic (I,V,L,F,M,W,Y,A) and polar (S,T,N,Q,R,H,K,D,E,C) residues")
        
        if seq_analysis['diversity_fraction'] >= 0.4:
            max_possible_score += 10.0
            triggered_conditions.append('diversity bonus')
            score += 10.0
            details['sequence_diversity'] = f"Excellent diversity: {seq_analysis['unique_residues']}/20 amino acids"
        elif seq_analysis['diversity_fraction'] >= 0.25:
            score += 5.0
            details['sequence_diversity'] = f"Good diversity: {seq_analysis['unique_residues']}/20 amino acids"
            recommendations.append("Increase amino acid diversity")
        else:
            details['sequence_diversity'] = f"Limited diversity: {seq_analysis['unique_residues']}/20 amino acids"
            recommendations.append("Increase amino acid diversity")
            
        normalized_score = (score / max_possible_score) * 100.0
        score = normalized_score
        details['score_normalization'] = f"Score normalized based on triggered conditions: {', '.join(triggered_conditions)}"
        
        evaluation['score'] = score
        evaluation['details'] = details
        evaluation['recommendations'] = recommendations
        
        if dist_matrix_analysis:
            from distance_matrix_analysis import summarize_distance_matrix_analysis, get_structural_recommendations
            
            position_context = {}
            for i in range(len(sequence)):
                context = {
                    'residue': sequence[i],
                    'position': i,
                    'structural_summary': summarize_distance_matrix_analysis(dist_matrix_analysis, i),
                    'recommendations': get_structural_recommendations(dist_matrix_analysis, i, design_goal_str),
                    'compactness': dist_matrix_analysis.get('compactness', {}).get(i, 'unknown'),
                    'contact_density': dist_matrix_analysis.get('contact_density', {}).get(i, 0),
                    'secondary_structure_hint': dist_matrix_analysis.get('secondary_structure_region', {}).get(i, 'unknown'),
                    'flexibility': dist_matrix_analysis.get('structural_flexibility', {}).get(i, 'unknown')
                }
                position_context[i] = context
            
            evaluation['position_specific_context'] = position_context
        
        if score >= 80:
            evaluation['goal_match'] = True
            evaluation['assessment'] = "EXCELLENT: Strong match to design goal"
        elif score >= 60:
            evaluation['goal_match'] = True  
            evaluation['assessment'] = "GOOD: Clear alignment with design goal"
        elif score >= 40:
            evaluation['assessment'] = "PARTIAL: Some features match design goal"
        else:
            evaluation['assessment'] = "POOR: Limited match to design goal"
        
    except Exception as e:
        evaluation['error'] = str(e)
        evaluation['assessment'] = f"✗ ERROR: Evaluation failed: {e}"
    
    return evaluation


def print_structure_evaluation(evaluation: Dict):
    """
    Print a formatted structure evaluation report.
    
    Args:
        evaluation: Evaluation dictionary from evaluate_design_goal()
    """
    print("\n" + "="*80)
    print("📊 STRUCTURE EVALUATION REPORT")
    print("="*80)
    
    print(f"Design Goal: {evaluation.get('design_goal', 'N/A')}")
    print(f"Sequence: {evaluation['sequence']}")
    print(f"Assessment: {evaluation['assessment']}")
    print(f"Score: {evaluation['score']:.1f}/100")
    
    if 'structure_analysis' in evaluation:
        sa = evaluation['structure_analysis']
        print(f"\n🔍 Structure Analysis:")
        print(f"  Length: {sa['length']} residues")
        print(f"  Secondary structure: {sa['ss_string']}")
        print(f"  Helices: {sa['num_helices']} regions {sa['helix_regions']}")
        print(f"  Sheets: {sa['num_sheets']} regions {sa['sheet_regions']}")
        print(f"  Loops: {sa['num_loops']} regions {sa['loop_regions']}")
    
    if 'sequence_analysis' in evaluation:
        seq = evaluation['sequence_analysis']
        print(f"\n🧬 Sequence Analysis:")
        print(f"  Hydrophobic: {seq['hydrophobic_fraction']:.1%}")
        print(f"  Polar: {seq['polar_fraction']:.1%}")
        print(f"  Charged: {seq['charged_fraction']:.1%}")
        print(f"  Flexible: {seq['flexible_fraction']:.1%}")
        print(f"  Diversity: {seq['unique_residues']}/20 amino acids")
    
    if 'details' in evaluation:
        print(f"\n📋 Detailed Evaluation:")
        for key, value in evaluation['details'].items():
            print(f"  {key}: {value}")
    
    if 'recommendations' in evaluation and evaluation['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in evaluation['recommendations']:
            print(f"  • {rec}")
    
    print("="*60)
