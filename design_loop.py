""" Design loop with 4 phases."""

import sys
import os
import shutil
import gc
import psutil
from io import StringIO
import pandas as pd
from tqdm import tqdm
from config import DesignConfig, DesignGrid
from structure_utils import compute_ca_distance_matrix, get_spatial_neighbors, get_exposure
from folding_utils import fold_with_omegafold, visualize_pdb, log_sequence, render_protein_with_pymol
from llm_interface import call_llm_agent
import rosetta_energy_utils as reu 
from structure_evaluation import evaluate_design_goal
from distance_matrix_analysis import analyze_distance_matrix
from memory_system import MemoryManager

def log_memory_usage(stage, iteration=None, memory_manager=None):
    """Log current memory usage for debugging OOM issues."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    
    prefix = f"🧠 MEM[{iteration}]" if iteration is not None else "🧠 MEM"
    
    mem_stats = ""
    if memory_manager:
        stats = memory_manager.get_memory_statistics()
        mem_stats = f" | Agents:{stats['local_histories']['tracked_agents']}/{stats['local_histories']['max_agents']} Actions:{stats['local_histories']['total_actions']} Buffer:{stats['local_histories']['current_actions_buffer']}"
    
    print(f"{prefix} {stage}: {memory_gb:.2f} GB{mem_stats}")
    
    if memory_gb > 100:
        print(f"🚨 EMERGENCY: Memory usage too high ({memory_gb:.2f}GB)!")
        if memory_manager:
            memory_manager._aggressive_cleanup()
        gc.collect()
        print(f"🚨 After emergency cleanup: {psutil.Process().memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
    elif memory_gb > 60:
        print(f"⚠️ High memory usage ({memory_gb:.2f}GB) - forcing garbage collection")
        gc.collect()
        
    return memory_gb


_DESIGN_LOOP_RUNNING = False

def run_design_loop(start_seq: str, config: DesignConfig):
    """Run the iterative protein sequence optimization loop with integrated learning memory."""
    global _DESIGN_LOOP_RUNNING
    
    if _DESIGN_LOOP_RUNNING:
        print("🚨 WARNING: run_design_loop called recursively! Blocking to prevent infinite loop.")
        return [], pd.DataFrame(), None
    
    _DESIGN_LOOP_RUNNING = True
    
    try:
        log_memory_usage("Starting design loop")
        grid = DesignGrid(start_seq)
        metadata = []
        rejected_sequences = []
        best_energy = float('inf')
        best_sequence = None
        
        memory_manager = MemoryManager(log_dir=os.path.join(config.log_dir, "memory_logs"))
        print(f"🧠 Memory system initialized (saving disabled). Previous stats: {memory_manager.get_memory_statistics()}")
        log_memory_usage("After initialization", memory_manager=memory_manager)

        i = 0
        sequence = grid.get_sequence()
        
        pdb_path = fold_with_omegafold(sequence, output_dir=f"{config.log_dir}/tmp_iter_{i}")
        
        if pdb_path is None or not os.path.exists(pdb_path):
            print("✗ Error: Initial structure prediction failed - no PDB file generated")
            print(f"  Sequence: {sequence}")
            
            final_pdb_path = os.path.join(config.log_dir, f"struct_iter_{i}.pdb")
            with open(final_pdb_path, 'w') as f:
                f.write("")
            
            dist_matrix = None
            config.current_dist_matrix = None
            config.current_pdb_path = final_pdb_path
            rendered_img_path = os.path.join(config.log_dir, f"image_iter_{i}.png")
            
        else:
            dist_matrix = compute_ca_distance_matrix(pdb_path)
            config.current_dist_matrix = dist_matrix
            
            final_pdb_path = os.path.join(config.log_dir, f"struct_iter_{i}.pdb")
            shutil.copy2(pdb_path, final_pdb_path)
            
            log_sequence(config.log_dir, i, sequence, final_pdb_path)
            config.current_pdb_path = final_pdb_path
            visualize_pdb(final_pdb_path)
            rendered_img_path = os.path.join(config.log_dir, f"image_iter_{i}.png")

        energies = reu.calculate_rosetta_energies(final_pdb_path, per_residue=False)
        energy_data = {}
        if energies:
            energy_data = {
                "total_energy": energies.get('total_energy', 0),
                "vdw_atr": energies.get('vdw_atr', 0),
                "vdw_rep": energies.get('vdw_rep', 0),
                "hbond": energies.get('hbond', 0),
                "score": energies.get('score', 0),
                "energy_change": 0,
                "energy_change_from_initial": 0
            }
            best_energy = energy_data['total_energy']
            best_sequence = sequence
            print(f"Initial energy - Total: {energy_data['total_energy']:.3f}, Score: {energy_data['score']:.3f}")
        else:
            print("Warning: Energy calculation failed for initial structure")
            energy_data = {
                "total_energy": float('inf'),
                "energy_change": 0,
                "energy_change_from_initial": 0
            }

        iteration_data = {
            "iteration": i,
            "sequence": sequence,
            "pdb_file": final_pdb_path,
            "rendered_image": rendered_img_path,
            "is_best_energy": True
        }
        iteration_data.update(energy_data)
        metadata.append(iteration_data)

        for i in range(1, config.max_iterations + 1):
            print(f"\nIteration {i}/{config.max_iterations}")
            log_memory_usage("Starting iteration", i, memory_manager)

            memory_manager.start_iteration(i-1, config.design_goal)

            try:
                original_sequence = grid.get_sequence()
                
                structure_feedback = None
                if metadata and 'structure_feedback' in metadata[-1]:
                    structure_feedback = metadata[-1]['structure_feedback']

                print("PHASE 1: Collecting agent proposals...")
                
                agent_processing_flag_file = "/tmp/swarm_agent_processing.flag"
                
                with open(agent_processing_flag_file, 'w') as f:
                    f.write("Agent processing in progress")
                
                try:
                    proposed_changes = []
                    for index in tqdm(grid.positions(), desc="Agent reasoning", leave=False):
                        spatial_neighbors = None
                        exposure = None
                        if config.current_dist_matrix is not None and config.current_dist_matrix.size > 0:
                            spatial_neighbors = get_spatial_neighbors(index, grid.get_sequence(), config.current_dist_matrix)
                            if spatial_neighbors is not None:
                                for neighbor in spatial_neighbors:
                                    neighbor_pos = neighbor['position']
                                    distance = config.current_dist_matrix[index, neighbor_pos]
                                    neighbor['distance'] = distance
                                spatial_neighbors.sort(key=lambda x: x['distance'])
                            
                            exposure = get_exposure(index, grid.get_sequence(), config.current_dist_matrix)
                        
                        position_structural_context = None
                        position_secondary_structure = None
                        if structure_feedback and 'position_specific_context' in structure_feedback and structure_feedback['position_specific_context'] is not None:
                            position_structural_context = structure_feedback['position_specific_context'].get(index)
                        
                        if index % 20 == 0:
                            current_mem = log_memory_usage(f"Agent processing pos {index}", i, memory_manager)
                            if current_mem and current_mem > 100:
                                print(f"⚠️ Memory critical during agent processing - forcing cleanup")
                                gc.collect()
                        
                        if structure_feedback and 'secondary_structure_analysis' in structure_feedback:
                            ss_analysis = structure_feedback['secondary_structure_analysis']
                            if 'structure_assignment' in ss_analysis and index < len(ss_analysis['structure_assignment']):
                                position_secondary_structure = {
                                    'dssp_assignment': ss_analysis['structure_assignment'][index],
                                    'dssp_available': ss_analysis.get('dssp_available', False),
                                }
                        
                        agent_input = {
                            'position': index,
                            'state': grid.sequence[index],
                            'sequence': grid.get_sequence(),
                            'neighbors': grid.get_neighbors(index),
                            'spatial_neighbors': spatial_neighbors,
                            'exposure': exposure,
                            'structural_context': position_structural_context, 
                            'secondary_structure': position_secondary_structure,
                            'goal': config.design_goal,
                        }
                        
                        memory_context = memory_manager.get_agent_memory_context(index, agent_input)
                        agent_input['memory_context'] = memory_context
                        
                        old_state = grid.sequence[index]
                        new_state = call_llm_agent(
                            agent_input, 
                            config=config, 
                            rejected_sequences=rejected_sequences,
                            structure_feedback=structure_feedback
                        )
                        
                        memory_manager.record_agent_action(index, old_state, new_state, agent_input)
                        
                        proposed_changes.append((index, new_state))
                finally:
                    if os.path.exists(agent_processing_flag_file):
                        os.remove(agent_processing_flag_file)

                print("PHASE 2: Applying all proposed changes...")
                for index, new_state in proposed_changes:
                    grid.update(index, new_state)

                sequence = grid.get_sequence()
                print(f"Proposed sequence {i}: {sequence}")
                
                sys.stdout.flush()
                sys.stderr.flush()

                print("\n" + "="*80)
                print("PHASE 3: Computing structure and energy for complete iteration...")
                print("="*80)
                
                pdb_path = fold_with_omegafold(sequence, output_dir=f"{config.log_dir}/tmp_iter_{i}")
                print(f"Proposed sequence {i}: {sequence}")
                
                if pdb_path is None or not os.path.exists(pdb_path):
                    print(f"✗ Error: Structure prediction failed for iteration {i}")
                    print(f"  Sequence: {sequence}")
                    print("  Reverting to previous sequence and skipping this iteration")
                    
                    grid.set_sequence(original_sequence)
                    sequence = original_sequence
                    memory_manager.record_iteration_result(i-1, False, "Structure prediction failed")
                    continue
                
                dist_matrix = compute_ca_distance_matrix(pdb_path)
                
                if dist_matrix is None:
                    print(f"✗ Error: Distance matrix computation failed for iteration {i}")
                    print("  Reverting to previous sequence and skipping this iteration")
                    
                    grid.set_sequence(original_sequence)
                    sequence = original_sequence
                    memory_manager.record_iteration_result(i-1, False, "Distance matrix computation failed")
                    continue
                
                energies = reu.calculate_rosetta_energies(pdb_path, per_residue=False)
                
                if energies:
                    energy_data = {
                        "total_energy": energies.get('total_energy', 0),
                        "vdw_atr": energies.get('vdw_atr', 0),
                        "vdw_rep": energies.get('vdw_rep', 0),
                        "hbond": energies.get('hbond', 0),
                        "score": energies.get('score', 0),
                    }
                    
                    if metadata and metadata[-1] is not None:
                        prev_energy = metadata[-1].get('total_energy', 0)
                        energy_data['energy_change'] = energy_data['total_energy'] - prev_energy
                        if metadata[0] is not None:
                            energy_data['energy_change_from_initial'] = energy_data['total_energy'] - metadata[0].get('total_energy', 0)
                        else:
                            energy_data['energy_change_from_initial'] = 0
                    else:
                        energy_data['energy_change'] = 0
                        energy_data['energy_change_from_initial'] = 0
                    
                    print(f"Proposed energy - Total: {energy_data['total_energy']:.3f} (Δ{energy_data['energy_change']:+.3f})")
                    print(f"Best energy so far - Total: {best_energy:.3f}")
                else:
                    print("✗ Warning: Energy calculation failed")
                    energy_data = {
                        "total_energy": float('inf'),
                        "energy_change": float('inf'),
                        "energy_change_from_initial": float('inf')
                    }

                print("Evaluating structure and making decision...")
                
                structure_evaluation = evaluate_design_goal(config.design_goal, pdb_path, sequence, dist_matrix)
                goal_match_score = structure_evaluation.get('score', 0)
                print(f"Structure evaluation: {structure_evaluation.get('assessment', 'UNKNOWN')}")
                print(f"Goal match score: {goal_match_score:.1f}/100")

                energy_improvement = best_energy - energy_data.get('total_energy', float('inf'))
                structure_improvement = goal_match_score - (metadata[-1].get('goal_match_score', 0) if metadata and metadata[-1] is not None else 0)
                current_energy = energy_data.get('total_energy', float('inf'))
                
                same_score_energy_improvement = False
                if abs(structure_improvement) <= 0.5 and metadata:
                    current_score = goal_match_score
                    same_score_energies = []
                    for m in metadata:
                        if m.get('goal_match_score') is not None and abs(m.get('goal_match_score', 0) - current_score) <= 0.5:
                            if m.get('total_energy') is not None and m.get('total_energy') != float('inf'):
                                same_score_energies.append(m.get('total_energy'))
                    
                    if same_score_energies:
                        best_same_score_energy = min(same_score_energies)
                        if current_energy < best_same_score_energy:
                            same_score_energy_improvement = True

                if energy_improvement > 0:
                    print(f"✅ ACCEPTED: Energy improved by {energy_improvement:.3f} (new best overall)")
                    accepted = True
                    best_energy = energy_data['total_energy']
                    best_sequence = sequence
                    config.current_dist_matrix = dist_matrix
                elif structure_improvement > 0:
                    print(f"✅ ACCEPTED: Structure goal progress (+{structure_improvement:.1f} points)")
                    accepted = True
                    config.current_dist_matrix = dist_matrix
                elif same_score_energy_improvement:
                    energy_gain = min([e for e in same_score_energies if e != float('inf')]) - current_energy
                    print(f"✅ ACCEPTED: Better energy for same structure score (improved by {energy_gain:.3f})")
                    accepted = True
                    config.current_dist_matrix = dist_matrix
                else:
                    print(f"❌ REJECTED: Energy worse by {abs(energy_improvement):.3f}, Structure score: {goal_match_score:.1f} (Δ{structure_improvement:+.1f})")
                    accepted = False
                    rejected_sequences.append(sequence)
                    
                    proposed_energy_data = energy_data.copy()
                    proposed_sequence = sequence
                    
                    grid.sequence = list(original_sequence)
                    sequence = original_sequence
                    
                    if metadata and metadata[-1] is not None:
                        final_energy_data = metadata[-1].copy()
                        energy_data = {
                            "total_energy": final_energy_data.get('total_energy', 0),
                            "vdw_atr": final_energy_data.get('vdw_atr', 0),
                            "vdw_rep": final_energy_data.get('vdw_rep', 0),
                            "hbond": final_energy_data.get('hbond', 0),
                            "score": final_energy_data.get('score', 0),
                            "energy_change": 0,
                            "energy_change_from_initial": final_energy_data.get('energy_change_from_initial', 0),
                            "proposed_energy": proposed_energy_data.get('total_energy', float('inf')),
                            "proposed_energy_change": proposed_energy_data.get('energy_change', 0),
                            "proposed_energy_breakdown": {
                                "vdw_atr": proposed_energy_data.get('vdw_atr', 0),
                                "vdw_rep": proposed_energy_data.get('vdw_rep', 0),
                                "hbond": proposed_energy_data.get('hbond', 0),
                                "score": proposed_energy_data.get('score', 0)
                            }
                        }
                    else:
                        energy_data = {
                            "total_energy": float('inf'),
                            "proposed_energy": proposed_energy_data.get('total_energy', float('inf')),
                            "energy_change": 0,
                            "energy_change_from_initial": 0
                        }
                    
                    pdb_path = metadata[-1]['pdb_file'] if metadata and metadata[-1] is not None and 'pdb_file' in metadata[-1] else pdb_path

                final_pdb_path = os.path.join(config.log_dir, f"struct_iter_{i}.pdb")
                if accepted:
                    shutil.copy2(pdb_path, final_pdb_path)
                else:
                    if metadata and metadata[-1] is not None and 'pdb_file' in metadata[-1]:
                        shutil.copy2(metadata[-1]['pdb_file'], final_pdb_path)
                
                config.current_pdb_path = final_pdb_path
                visualize_pdb(final_pdb_path)
                
                rendered_img_path = os.path.join(config.log_dir, f"image_iter_{i}.png")
                success = render_protein_with_pymol(final_pdb_path, rendered_img_path)
                if success:
                    print(f"🖼️  Structure image saved: {rendered_img_path}")
                else:
                    print(f"⚠️  Warning: Failed to render structure image for iteration {i}")

                iteration_data = {
                    "iteration": i,
                    "sequence": sequence,
                    "pdb_file": final_pdb_path,
                    "rendered_image": rendered_img_path,
                    "structure_feedback": structure_evaluation,
                    "goal_match_score": goal_match_score,
                    "accepted": accepted,
                    "is_best_energy": sequence == best_sequence,
                    "best_energy_so_far": best_energy,
                    "best_sequence_so_far": best_sequence,
                    "energy_improvement": energy_improvement,
                    "structure_improvement": structure_improvement
                }
                iteration_data.update(energy_data)
                metadata.append(iteration_data)

                print(f"\n📊 ENERGY PROGRESSION SUMMARY:")
                print(f"   Current energy: {energy_data.get('total_energy', 'N/A'):.3f}")
                print(f"   Change from previous: {energy_data.get('energy_change', 0):+.3f}")
                print(f"   Change from initial: {energy_data.get('energy_change_from_initial', 0):+.3f}")
                print(f"   Best energy so far: {best_energy:.3f}")
                print(f"   Structure score: {goal_match_score:.1f}/100 (Δ{structure_improvement:+.1f})")
                
                if metadata:
                    current_score = goal_match_score
                    same_score_energies = []
                    for m in metadata:
                        if m.get('goal_match_score') is not None and abs(m.get('goal_match_score', 0) - current_score) <= 0.5:
                            if m.get('total_energy') is not None and m.get('total_energy') != float('inf'):
                                same_score_energies.append({
                                    'energy': m.get('total_energy'),
                                    'iteration': m.get('iteration', 0),
                                    'accepted': m.get('accepted', False)
                                })
                    
                    if same_score_energies:
                        best_same_score = min(same_score_energies, key=lambda x: x['energy'])
                        current_energy = energy_data.get('total_energy', float('inf'))
                        
                        if best_same_score['energy'] < current_energy:
                            energy_gap = current_energy - best_same_score['energy']
                            print(f"   🎯 Best energy with score {current_score:.1f}: {best_same_score['energy']:.3f} (iter {best_same_score['iteration']})")
                            print(f"   📈 Energy gap to best same score: +{energy_gap:.3f} (room for improvement)")
                        elif best_same_score['energy'] == current_energy:
                            print(f"   🏆 This is the best energy achieved with score {current_score:.1f}!")
                        else:
                            print(f"   🎯 Previous best energy with score {current_score:.1f}: {best_same_score['energy']:.3f} (iter {best_same_score['iteration']})")
                            print(f"   🎉 New best energy for this structure score!")
                
                if accepted:
                    acceptance_rate = sum(1 for m in metadata if m.get('accepted', False)) / len(metadata) * 100
                    print(f"   ✅ Acceptance rate: {acceptance_rate:.1f}% ({sum(1 for m in metadata if m.get('accepted', False))}/{len(metadata)} iterations)")
                else:
                    print(f"   ❌ Rejected - keeping previous structure")

                if energy_improvement > 0 and structure_improvement > 0:
                    reason = "energy and structure improved"
                elif energy_improvement > 0:
                    reason = "energy improved"
                elif structure_improvement > 0:
                    reason = "structure improved"
                elif energy_improvement < 0 and structure_improvement < 0:
                    reason = "energy and structure worsened"
                elif energy_improvement < 0:
                    reason = "energy worsened"
                elif structure_improvement < 0:
                    reason = "structure worsened"
                else:
                    reason = "no improvement"
                
                if not accepted and 'proposed_sequence' in locals():
                    actual_proposed_sequence = proposed_sequence
                else:
                    actual_proposed_sequence = grid.get_sequence()
                
                memory_manager.finalize_iteration(
                    initial_sequence=original_sequence,
                    proposed_sequence=actual_proposed_sequence,
                    final_sequence=sequence,
                    accepted=accepted,
                    reason=reason,
                    energy_data=energy_data,
                    structure_score=goal_match_score
                )

            except Exception as e:
                print(f"Error in iteration {i}: {str(e)}")
                continue
            finally:
                log_memory_usage("Before iteration cleanup", i)
                
                if 'dist_matrix' in locals():
                    del dist_matrix
                if 'proposed_changes' in locals():
                    del proposed_changes
                if 'energies' in locals():
                    del energies
                if 'pdb_path' in locals() and pdb_path and os.path.exists(pdb_path):
                    import glob
                    old_pdbs = glob.glob(f"{config.log_dir}/tmp_iter_*.pdb")
                    for old_pdb in old_pdbs:
                        iter_num = int(old_pdb.split('_iter_')[1].split('.pdb')[0])
                        if iter_num < i - 1:
                            os.remove(old_pdb)
                
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    for _ in range(3):
                        torch.cuda.empty_cache()
                
                for _ in range(3):
                    gc.collect()
                    
                log_memory_usage("After iteration cleanup", i)
                
                current_mem = log_memory_usage("Iteration complete", i)
                
                mem_limit_gb = 850
                import subprocess
                result = subprocess.run(['scontrol', 'show', 'job', os.environ.get('SLURM_JOB_ID', '')], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'MaxMemory=' in line:
                            mem_str = line.split('MaxMemory=')[1].split()[0]
                            if mem_str.endswith('G'):
                                mem_limit_gb = int(mem_str[:-1])
                            elif mem_str.endswith('M'):
                                mem_limit_gb = int(mem_str[:-1]) / 1024
                            break
                
                mem_threshold = mem_limit_gb * 0.95
                
                if current_mem and current_mem > mem_threshold:
                    print(f"🚨 CRITICAL MEMORY USAGE ({current_mem:.2f}GB) exceeds {mem_threshold:.0f}GB threshold ({mem_threshold/mem_limit_gb*100:.0f}% of {mem_limit_gb}GB) - STOPPING EARLY")
                    break

        return metadata, pd.DataFrame(metadata), memory_manager
    
    finally:
        _DESIGN_LOOP_RUNNING = False 