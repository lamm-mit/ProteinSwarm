#!/usr/bin/env python3
"""Comprehensive test script for multiple design goals."""

import os
import sys
import json
import time
import traceback
from datetime import datetime
import logging
import argparse

os.environ['TQDM_DISABLE'] = 'False'
if 'JUPYTER_CONFIG_DIR' in os.environ:
    del os.environ['JUPYTER_CONFIG_DIR']

from config import DesignConfig
from design_loop import run_design_loop

DESIGN_GOALS = [
    {
        "name": "alpha_helices_hydrophilic",
        "goal": "form local alpha helices using hydrophilic residues like serine, threonine, aspartate, glutamate, and others",
        "folder": "test_alpha_helices_hydrophilic",
        "start_seq": "S"*18
    },
    {
        "name": "loose_coils",
        "goal": "form loose, extended coils using polar and charged residues to reduce compaction",
        "folder": "test_loose_coils", 
        "start_seq": "L"*10
    },
    {
        "name": "beta_strands",
        "goal": "form beta strands by placing alternating hydrophobic and polar residues",
        "folder": "test_beta_strands", 
        "start_seq": "A"*20
    },
    {
        "name": "alpha_helices_alanine_leucine_glutamate",
        "goal": "form alpha helices using alanine, leucine, and glutamate in repeating patterns",
        "folder": "test_alpha_helices_alanine_leucine_glutamate",
        "start_seq": "S"*21
    },

 ]

def setup_logging(log_dir):
    """Set up logging for this test run."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('design_goals_test')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    log_file = os.path.join(log_dir, 'test_run.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def test_single_design_goal(goal_config, iterations=64, logger=None):
    """Test a single design goal."""
    if logger is None:
        logger = logging.getLogger('design_goals_test')
    
    start_time = time.time()
    
    logger.info(f"🎯 Starting test: {goal_config['name']}")
    logger.info(f"   Goal: {goal_config['goal']}")
    logger.info(f"   Start sequence: {goal_config['start_seq']}")
    logger.info(f"   Output folder: {goal_config['folder']}")
    
    try:
        config = DesignConfig(
            system_description=(
                "You are an agent participating in the design of a linear protein sequence, "
                "represented in FASTA format using one-letter amino acid codes. "
                "Each agent is responsible for optimizing one residue in the sequence. "
                "Work collaboratively toward the global design goal."
            ),
            design_goal=goal_config['goal'],
            max_iterations=iterations,
            neighbor_radius=10,
            log_dir=goal_config['folder'],
            visualization=True
        )
        
        os.makedirs(config.log_dir, exist_ok=True)
        
        goal_info = {
            "name": goal_config['name'],
            "goal": goal_config['goal'],
            "start_sequence": goal_config['start_seq'],
            "iterations": iterations,
            "start_time": datetime.now().isoformat(),
            "test_purpose": "Evaluate LLM agent performance on specific structural design objectives"
        }
        
        with open(os.path.join(config.log_dir, 'goal_config.json'), 'w') as f:
            json.dump(goal_info, f, indent=2)
        
        logger.info(f"   Running {iterations} iterations...")
        metadata, df, _ = run_design_loop(goal_config['start_seq'], config)
        
        results_file = os.path.join(config.log_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        if df is not None:
            df_file = os.path.join(config.log_dir, 'results.csv')
            df.to_csv(df_file, index=False)
        
        elapsed_time = time.time() - start_time
        
        final_sequence = metadata[-1]['sequence'] if metadata else goal_config['start_seq']
        logger.info(f"✅ Completed {goal_config['name']} in {elapsed_time:.1f}s")
        logger.info(f"   Final sequence: {final_sequence}")
        
        if metadata and len(metadata) > 1:
            initial_energy = metadata[0].get('total_energy', 'N/A')
            final_energy = metadata[-1].get('total_energy', 'N/A')
            logger.info(f"   Energy change: {initial_energy} → {final_energy}")
            
            if 'structure_score' in metadata[-1]:
                final_score = metadata[-1]['structure_score']
                logger.info(f"   Final structure score: {final_score}/100")
        
        goal_info.update({
            "end_time": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "final_sequence": final_sequence,
            "success": True,
            "iterations_completed": len(metadata) - 1 if metadata else 0
        })
        
        with open(os.path.join(config.log_dir, 'goal_config.json'), 'w') as f:
            json.dump(goal_info, f, indent=2)
        
        return True, goal_config['name'], elapsed_time, final_sequence
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"❌ Failed {goal_config['name']}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        try:
            error_info = goal_info.copy() if 'goal_info' in locals() else goal_config.copy()
            error_info.update({
                "end_time": datetime.now().isoformat(),
                "elapsed_time_seconds": elapsed_time,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            os.makedirs(goal_config['folder'], exist_ok=True)
            with open(os.path.join(goal_config['folder'], 'error_info.json'), 'w') as f:
                json.dump(error_info, f, indent=2)
        except:
            pass
        
        return False, goal_config['name'], elapsed_time, str(e)

def run_all_design_goals(iterations=64):
    """Run all design goals in sequence."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_dir = f"design_goals_test_{timestamp}"
    
    logger = setup_logging(master_log_dir)
    
    logger.info("🚀 Starting comprehensive design goals test")
    logger.info(f"   Master log directory: {master_log_dir}")
    logger.info(f"   Number of goals: {len(DESIGN_GOALS)}")
    logger.info(f"   Iterations per goal: {iterations}")
    logger.info("=" * 80)
    
    overall_start = time.time()
    results = []
    
    for i, goal_config in enumerate(DESIGN_GOALS, 1):
        logger.info(f"[{i}/{len(DESIGN_GOALS)}] Testing goal: {goal_config['name']}")
        
        goal_dir = os.path.join(master_log_dir, goal_config['folder'])
        goal_config_copy = goal_config.copy()
        goal_config_copy['folder'] = goal_dir
        
        success, name, elapsed, result = test_single_design_goal(
            goal_config_copy, iterations, logger
        )
        
        results.append({
            "name": name,
            "success": success,
            "elapsed_time": elapsed,
            "result": result,
            "folder": goal_dir
        })
        
        logger.info("-" * 40)
    
    overall_elapsed = time.time() - overall_start
    successful = sum(1 for r in results if r['success'])
    
    logger.info("🏁 FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    logger.info(f"Successful tests: {successful}/{len(DESIGN_GOALS)}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        logger.info(f"  {status} {result['name']}: {result['elapsed_time']:.1f}s - {result['result']}")
    
    summary = {
        "timestamp": timestamp,
        "master_log_dir": master_log_dir,
        "total_elapsed_seconds": overall_elapsed,
        "successful_tests": successful,
        "total_tests": len(DESIGN_GOALS),
        "iterations_per_goal": iterations,
        "results": results
    }
    
    with open(os.path.join(master_log_dir, 'master_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n📁 All results saved in: {master_log_dir}")
    logger.info("Check individual goal folders for detailed logs and structures!")
    
    return summary

def main():
    """Main function for running design goal tests."""
    parser = argparse.ArgumentParser(description='Run design goals tests')
    parser.add_argument('iterations', type=int, nargs='?', default=5,
                       help='Number of iterations per goal (default: 5)')
    parser.add_argument('--goal', type=str, 
                       help='Run specific goal only (e.g., extended_beta)')
    
    args = parser.parse_args()
    iterations = args.iterations
    
    if args.goal:
        goal_config = None
        for config in DESIGN_GOALS:
            if config['name'] == args.goal:
                goal_config = config
                break
        
        if goal_config is None:
            print(f"❌ Goal '{args.goal}' not found!")
            print("Available goals:")
            for config in DESIGN_GOALS:
                print(f"  - {config['name']}")
            return 1
        
        print(f"🎯 Running single goal: {args.goal}")
        print(f"   Iterations: {iterations}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        goal_dir = f"{goal_config['folder']}_{timestamp}"
        goal_config_copy = goal_config.copy()
        goal_config_copy['folder'] = goal_dir
        
        logger = setup_logging(goal_dir)
        
        try:
            success, name, elapsed, result = test_single_design_goal(
                goal_config_copy, iterations, logger
            )
            
            if success:
                print(f"\n✅ Goal {name} completed successfully!")
                print(f"   Time: {elapsed:.1f}s")
                print(f"   Result: {result}")
                print(f"   Results in: {goal_dir}")
                return 0
            else:
                print(f"\n❌ Goal {name} failed: {result}")
                return 1
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Goal {args.goal} interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Critical error in goal {args.goal}: {e}")
            traceback.print_exc()
            return 1
    
    else:
        print(f"🚀 Running all {len(DESIGN_GOALS)} goals with {iterations} iterations each")
        
        try:
            summary = run_all_design_goals(iterations)
            print(f"\n✅ All tests completed! Results in: {summary['master_log_dir']}")
            return 0
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 