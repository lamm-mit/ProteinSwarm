"""Energy and score monitor."""

import re
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class EnergyMonitor:
    def __init__(self, log_file: str, csv_file: str = None, poll_interval: float = 1.0):
        """Initialize the energy monitor."""
        self.log_file = Path(log_file)
        self.poll_interval = poll_interval
        self.last_position = 0
        
        if csv_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = Path(f"energy_scores_{timestamp}.csv")
        else:
            self.csv_file = Path(csv_file)
            
        self.headers = [
            'timestamp', 'iteration', 'proposed_energy', 'energy_delta', 'best_energy', 
            'structure_score', 'goal_match', 'best_score', 'sequence', 
            'best_energy_sequence', 'best_score_sequence'
        ]
        
        self.best_energy = float('inf')
        self.best_energy_sequence = ""
        self.best_score = 0.0
        self.best_score_sequence = ""
        self.iteration_count = 1
        self.starting_sequence = ""
        
        self._init_csv()
        
        self.patterns = {
            'initial_energy': re.compile(r'Initial energy - Total: ([\d.]+), Score: ([\d.]+)'),
            'proposed_energy': re.compile(r'Proposed energy - Total: ([\d.]+)(?:\s*\(Δ([+-][\d.]+)\))?'),
            'best_energy': re.compile(r'Best energy so far - Total: ([\d.]+)'),
            'structure_score': re.compile(r'Structure score: ([\d.-]+)/100'),
            'goal_match': re.compile(r'Goal match: (NOT MATCHED|MATCHED)'),
            'start_sequence': re.compile(r'Start sequence: ([A-Z]+)'),
            'proposed_sequence': re.compile(r'Proposed sequence (\d+): ([A-Z]+)'),
            'assessment': re.compile(r'Assessment: [✗✓] ([A-Z]+):')
        }
        
        print(f"Monitoring: {self.log_file}")
        print(f"Output CSV: {self.csv_file}")
        print(f"Poll interval: {self.poll_interval}s")
        
    def _init_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        print(f"Initialized CSV file: {self.csv_file}")
    
    def _parse_log_chunk(self, chunk: str) -> List[Dict]:
        """Parse a chunk of log data and extract energy information."""
        entries = []
        lines = chunk.split('\n')
        current_entry = {}
        
        if not self.starting_sequence:
            for line in lines:
                start_seq_match = self.patterns['start_sequence'].search(line)
                if start_seq_match:
                    self.starting_sequence = start_seq_match.group(1)
                    break
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            initial_match = self.patterns['initial_energy'].search(line)
            if initial_match:
                if current_entry:
                    entries.append(self._finalize_entry(current_entry))
                
                initial_energy = float(initial_match.group(1))
                initial_score = float(initial_match.group(2))
                start_seq = self.starting_sequence if self.starting_sequence else ""
                
                current_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'iteration': 0,
                    'proposed_energy': initial_energy,
                    'energy_delta': 0.0,
                    'best_energy': initial_energy,
                    'structure_score': 0.0,
                    'goal_match': '',
                    'best_score': 0.0,
                    'sequence': start_seq,
                    'best_energy_sequence': start_seq,
                    'best_score_sequence': start_seq,
                }
                
                self.best_energy = initial_energy
                self.best_score = 0.0
                self.best_energy_sequence = start_seq
                self.best_score_sequence = start_seq
            
            energy_match = self.patterns['proposed_energy'].search(line)
            if energy_match:
                if current_entry:
                    entries.append(self._finalize_entry(current_entry))
                
                proposed_energy = float(energy_match.group(1))
                energy_delta = float(energy_match.group(2)) if energy_match.group(2) else 0.0
                
                current_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'iteration': self.iteration_count,
                    'proposed_energy': proposed_energy,
                    'energy_delta': energy_delta,
                    'best_energy': self.best_energy,
                    'structure_score': '',
                    'goal_match': '',
                    'best_score': self.best_score,
                    'sequence': '',
                    'best_energy_sequence': self.best_energy_sequence,
                    'best_score_sequence': self.best_score_sequence,
                }
                
                if proposed_energy < self.best_energy:
                    self.best_energy = proposed_energy
                current_entry['best_energy'] = self.best_energy
                
                self.iteration_count += 1
                
                for j in range(max(0, i-100), min(len(lines), i+100)):
                    score_match = self.patterns['structure_score'].search(lines[j])
                    if score_match and not current_entry['structure_score']:
                        current_entry['structure_score'] = float(score_match.group(1))
                    
                    goal_match = self.patterns['goal_match'].search(lines[j])
                    if goal_match and not current_entry['goal_match']:
                        current_entry['goal_match'] = goal_match.group(1)
                    
                    seq_match = self.patterns['proposed_sequence'].search(lines[j])
                    if seq_match:
                        current_entry['sequence'] = seq_match.group(2)
                        current_entry['iteration'] = int(seq_match.group(1))
                
                if current_entry.get('structure_score'):
                    structure_score = current_entry['structure_score']
                    if structure_score > self.best_score:
                        self.best_score = structure_score
                        if current_entry['sequence']:
                            self.best_score_sequence = current_entry['sequence']
                
                if current_entry['proposed_energy'] == self.best_energy and current_entry['sequence']:
                    self.best_energy_sequence = current_entry['sequence']
            
            best_energy_match = self.patterns['best_energy'].search(line)
            if best_energy_match and current_entry:
                best_energy = float(best_energy_match.group(1))
                if best_energy < self.best_energy:
                    self.best_energy = best_energy
                current_entry['best_energy'] = self.best_energy
        
        if current_entry:
            entries.append(self._finalize_entry(current_entry))
        
        return entries
    
    def _finalize_entry(self, entry: Dict) -> Dict:
        """Finalize an entry with current best values."""
        entry['best_energy'] = self.best_energy
        entry['best_score'] = self.best_score
        entry['best_energy_sequence'] = self.best_energy_sequence
        entry['best_score_sequence'] = self.best_score_sequence
        return entry
    
    def _reset_state(self):
        """Reset internal state for fresh parsing."""
        self.best_energy = float('inf')
        self.best_energy_sequence = ""
        self.best_score = 0.0
        self.best_score_sequence = ""
        self.iteration_count = 1
        self.starting_sequence = ""
    
    def _append_to_csv(self, entries: List[Dict]):
        """Append new entries to the CSV file."""
        if not entries:
            return
            
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            for entry in entries:
                writer.writerow(entry)
        
        print(f"Added {len(entries)} entries to CSV")
        for entry in entries[-3:]:
            iteration = entry['iteration']
            energy = entry['proposed_energy']
            score = entry['structure_score']
            goal = entry['goal_match']
            if iteration == 0:
                print(f"  Iter {iteration} (Initial): Energy: {energy} - Score: {score}")
            else:
                print(f"  Iter {iteration}: Energy: {energy} (Δ{entry['energy_delta']:+.1f}) - Score: {score} - Goal: {goal}")
        
        if entries:
            print(f"  Best Energy: {self.best_energy} - Best Score: {self.best_score}")
            print(f"  Best Energy Sequence: {self.best_energy_sequence}")
            print(f"  Best Score Sequence: {self.best_score_sequence}")
    
    def monitor(self):
        """Main monitoring loop."""
        while True:
            if not self.log_file.exists():
                print(f"Waiting for log file: {self.log_file}")
                time.sleep(self.poll_interval)
                continue
            
            current_size = self.log_file.stat().st_size
            
            if current_size > self.last_position:
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.last_position)
                    new_content = f.read()
                    self.last_position = current_size
                
                entries = self._parse_log_chunk(new_content)
                
                if entries:
                    self._append_to_csv(entries)
            
            time.sleep(self.poll_interval)
    
    def parse_existing(self):
        """Parse the entire existing log file."""
        if not self.log_file.exists():
            print(f"Log file not found: {self.log_file}")
            return
        
        print("Parsing existing log file...")
        self._reset_state()
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            self.last_position = len(content.encode('utf-8'))
        
        entries = self._parse_log_chunk(content)
        self._append_to_csv(entries)
        print(f"Parsed {len(entries)} entries from existing log")
        print(f"Final best energy: {self.best_energy}")
        print(f"Final best score: {self.best_score}")

def main():
    parser = argparse.ArgumentParser(description='Monitor protein design log for energy scores')
    parser.add_argument('log_file', help='Path to the log file to monitor')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('-i', '--interval', type=float, default=1.0, 
                       help='Polling interval in seconds (default: 1.0)')
    parser.add_argument('--parse-existing', action='store_true',
                       help='Parse existing log file content before monitoring')
    
    args = parser.parse_args()
    
    monitor = EnergyMonitor(args.log_file, args.output, args.interval)
    
    if args.parse_existing:
        monitor.parse_existing()
    
    print("Starting real-time monitoring... Press Ctrl+C to stop")
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\nStopping monitor...")

if __name__ == '__main__':
    main() 