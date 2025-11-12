"""Memory system."""

import json
import os
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from constants import DEFAULT_EXPOSURE_THRESHOLD, DEFAULT_NEIGHBORS


@dataclass
class ActionRecord:
    """Minimal record of agent action."""
    iteration: int
    position: int
    old_state: str
    new_state: str
    goal: str
    was_accepted: Optional[bool] = None
    energy_change: Optional[float] = None


@dataclass
class IterationOutcome:
    """Minimal iteration outcome record."""
    iteration: int
    accepted: bool
    reason: str
    energy: float
    structure_score: float
    mutation_count: int
    sequence: str 


class GlobalMemory:
    """Lightweight global memory with aggressive pruning."""
    
    def __init__(self, save_path: str = "global_memory.json"):
        self.save_path = save_path
        self.iteration_outcomes: List[IterationOutcome] = []
        self.global_design_goal: str = ""
        self.successful_patterns: Dict[str, int] = {}
        self.failed_patterns: Dict[str, int] = {}
        self.energy_trends: List[Tuple[int, float]] = []
        
        self.MAX_ITERATIONS = 5
        self.MAX_PATTERNS = 30
        self.EMERGENCY_ITERATIONS = 2
        
        self.total_iterations_run = 0
        self.total_iterations_accepted = 0
        self.cumulative_patterns: Dict[str, Tuple[int, int]] = {}
        self.best_energy_ever = float('inf')
        self.worst_energy_ever = float('-inf')
        self.energy_history_summary: List[Tuple[int, float, bool]] = []
    
    def update_goal(self, goal: str):
        """Update the global design goal."""
        self.global_design_goal = goal
    
    def add_iteration_outcome(self, outcome: IterationOutcome):
        """Add iteration outcome with immediate aggressive pruning."""
        self.iteration_outcomes.append(outcome)
        
        self.total_iterations_run += 1
        if outcome.accepted:
            self.total_iterations_accepted += 1
        
        if outcome.energy < self.best_energy_ever:
            self.best_energy_ever = outcome.energy
        if outcome.energy > self.worst_energy_ever:
            self.worst_energy_ever = outcome.energy
        
        self.energy_history_summary.append((outcome.iteration, outcome.energy, outcome.accepted))
        if len(self.energy_history_summary) > 50:
            self.energy_history_summary = self.energy_history_summary[-50:]
        
        memory_pressure = len(self.iteration_outcomes) > self.MAX_ITERATIONS * 2
        if memory_pressure:
            print(f"⚠️ Memory pressure detected, emergency cleanup!")
            self.iteration_outcomes = self.iteration_outcomes[-self.EMERGENCY_ITERATIONS:]
            self.successful_patterns = {}
            self.failed_patterns = {}
            gc.collect()
        else:
            if len(self.iteration_outcomes) > self.MAX_ITERATIONS:
                self.iteration_outcomes = self.iteration_outcomes[-self.MAX_ITERATIONS:]
        
        self.energy_trends.append((outcome.iteration, outcome.energy))
        if len(self.energy_trends) > self.MAX_ITERATIONS:
            self.energy_trends = self.energy_trends[-self.MAX_ITERATIONS:]
        
        self._prune_patterns()
        
        if outcome.iteration % 3 == 0:
            gc.collect()
    
    def _prune_patterns(self):
        """Aggressive pattern pruning."""
        if len(self.successful_patterns) > self.MAX_PATTERNS:
            sorted_patterns = sorted(self.successful_patterns.items(), key=lambda x: x[1], reverse=True)
            self.successful_patterns = dict(sorted_patterns[:self.MAX_PATTERNS])
            
        if len(self.failed_patterns) > self.MAX_PATTERNS:
            sorted_patterns = sorted(self.failed_patterns.items(), key=lambda x: x[1], reverse=True)
            self.failed_patterns = dict(sorted_patterns[:self.MAX_PATTERNS])
    

    
    def add_pattern_outcome(self, pattern: str, success: bool):
        """Add pattern outcome to cumulative tracking."""
        if pattern not in self.cumulative_patterns:
            self.cumulative_patterns[pattern] = (0, 0)
        
        successes, total = self.cumulative_patterns[pattern]
        total += 1
        if success:
            successes += 1
        self.cumulative_patterns[pattern] = (successes, total)
    
    def get_pattern_success_rate(self, pattern: str) -> float:
        """Get success rate for a specific mutation pattern from cumulative data."""
        if pattern in self.cumulative_patterns:
            successes, total = self.cumulative_patterns[pattern]
            return successes / total if total > 0 else 0.0
        
        successes = self.successful_patterns.get(pattern, 0)
        failures = self.failed_patterns.get(pattern, 0)
        total = successes + failures
        return successes / total if total > 0 else 0.0
    
    def get_global_insights(self) -> Dict[str, Any]:
        """Generate global insights using cumulative data."""
        total_iterations = self.total_iterations_run
        overall_acceptance_rate = self.total_iterations_accepted / total_iterations if total_iterations > 0 else 0
        
        recent = self.iteration_outcomes[-5:] if len(self.iteration_outcomes) >= 5 else self.iteration_outcomes
        recent_acceptance_rate = sum(1 for o in recent if o.accepted) / len(recent) if recent else 0
        
        energy_trend = "stable"
        if len(self.energy_history_summary) >= 10:
            recent_energies = [e[1] for e in self.energy_history_summary[-5:]]
            early_energies = [e[1] for e in self.energy_history_summary[:5]]
            
            recent_avg = sum(recent_energies) / len(recent_energies)
            early_avg = sum(early_energies) / len(early_energies)
            
            if recent_avg < early_avg - 50:
                energy_trend = "improving"
            elif recent_avg > early_avg + 50:
                energy_trend = "worsening"
        
        return {
            "total_iterations": total_iterations,
            "acceptance_rate": overall_acceptance_rate,
            "recent_acceptance_rate": recent_acceptance_rate,
            "energy_trend": energy_trend,
            "current_goal": self.global_design_goal,
            "best_energy_ever": self.best_energy_ever if self.best_energy_ever != float('inf') else None,
            "worst_energy_ever": self.worst_energy_ever if self.worst_energy_ever != float('-inf') else None
        }
    
    def save(self):
        """Save memory to disk - DISABLED for performance."""
        pass
    
    def load(self):
        """Load memory from disk."""
        if not os.path.exists(self.save_path):
            return
        
        with open(self.save_path, 'r') as f:
            data = json.load(f)
        
        self.accepted_sequences = data.get("accepted_sequences", [])
        self.rejected_sequences = data.get("rejected_sequences", [])
        self.global_design_goal = data.get("global_design_goal", "")
        self.successful_patterns = data.get("successful_patterns", {})
        self.failed_patterns = data.get("failed_patterns", {})
        self.energy_trends = data.get("energy_trends", [])
        self.structure_trends = data.get("structure_trends", [])
        
        outcomes_data = data.get("iteration_outcomes", [])
        self.iteration_outcomes = []
        for outcome_dict in outcomes_data:
            simplified_dict = {
                "iteration": outcome_dict.get("iteration", 0),
                "accepted": outcome_dict.get("accepted", False),
                "reason": outcome_dict.get("reason", "")[:50],
                "energy": outcome_dict.get("energy", 0),
                "structure_score": outcome_dict.get("structure_score", 0),
                "mutation_count": outcome_dict.get("mutation_count", 0),
                "sequence": outcome_dict.get("sequence", "N/A")
            }
            self.iteration_outcomes.append(IterationOutcome(**simplified_dict))


class LocalHistory:
    """
    Local history system for individual agents.
    Tracks each agent's actions, decisions, and their impacts over time.
    """
    
    def __init__(self, position: int, save_dir: str = "local_histories"):
        self.position = position
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, f"agent_{position}_history.json")
        
        self.action_history: List[ActionRecord] = []
        self.success_count = 0
        self.failure_count = 0
        self.personal_patterns: Dict[str, float] = {}
        
        self.MAX_ACTIONS = 10
        self.MAX_PATTERNS = 15
        self.EMERGENCY_ACTIONS = 3
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.load()
    
    def add_action(self, action: ActionRecord):
        """Add an action with immediate aggressive pruning."""
        self.action_history.append(action)
        
        if action.was_accepted is not None:
            if action.was_accepted:
                self.success_count += 1
            else:
                self.failure_count += 1
        
        if action.old_state != action.new_state:
            pattern = f"{action.old_state}->{action.new_state}"
            self._update_personal_pattern_success(pattern, action.was_accepted)
        
        self._prune_local_memory()
        
        self.save()
    
    def _prune_local_memory(self):
        """Aggressive local memory pruning with emergency limits."""
        if len(self.action_history) > self.MAX_ACTIONS * 2:
            print(f"⚠️ Agent {self.position}: Emergency memory cleanup!")
            self.action_history = self.action_history[-self.EMERGENCY_ACTIONS:]
            self.personal_patterns = {}
            gc.collect()
        elif len(self.action_history) > self.MAX_ACTIONS:
            self.action_history = self.action_history[-self.MAX_ACTIONS:]
        
        if len(self.personal_patterns) > self.MAX_PATTERNS:
            sorted_patterns = sorted(self.personal_patterns.items(), key=lambda x: x[1], reverse=True)
            self.personal_patterns = dict(sorted_patterns[:self.MAX_PATTERNS])
    
    def _update_personal_pattern_success(self, pattern: str, success: Optional[bool]):
        """Update personal success rate for a pattern."""
        if success is None:
            return
        
        current_rate = self.personal_patterns.get(pattern, 0.5)
        learning_rate = 0.3
        new_rate = current_rate * (1 - learning_rate) + (1.0 if success else 0.0) * learning_rate
        self.personal_patterns[pattern] = new_rate
    
    def get_personal_insights(self) -> Dict[str, Any]:
        """Generate insights about this agent's performance and patterns."""
        recent_actions = self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
        
        insights = {
            "position": self.position,
            "total_actions": len(self.action_history),
            "overall_success_rate": self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0,
            "recent_success_rate": sum(1 for action in recent_actions if action.was_accepted) / len(recent_actions) if recent_actions else 0,
            "preferred_mutations": sorted([(pattern, rate) for pattern, rate in self.personal_patterns.items() if rate > 0.7], key=lambda x: x[1], reverse=True),
            "avoided_mutations": sorted([(pattern, rate) for pattern, rate in self.personal_patterns.items() if rate < 0.3], key=lambda x: x[1]),
            "recent_trend": self._analyze_recent_trend()
        }
        
        return insights
    
    def _analyze_recent_trend(self) -> str:
        """Analyze recent performance trend."""
        if len(self.action_history) < 10:
            return None
        
        recent_10 = self.action_history[-10:]
        recent_successes = sum(1 for action in recent_10 if action.was_accepted)
        recent_successes_rate = recent_successes / len(recent_10)
        
        if recent_successes_rate >= 0.7:
            return "improving"
        elif recent_successes_rate <= 0.3:
            return "declining"
        else:
            return "not better or worse"
    
    def get_relevant_experience(self, current_context: Dict[str, Any]) -> List[ActionRecord]:
        """Get relevant past actions based on current context."""
        current_state = current_context.get('state', '')
        current_neighbors = current_context.get('neighbors', {})
        
        relevant_actions = []
        for action in self.action_history:
            similarity_score = 0
            
            if action.old_state == current_state:
                similarity_score += 3
            
            try:
                if isinstance(current_neighbors, dict):
                    current_neighbor_values = set(current_neighbors.values())
                elif isinstance(current_neighbors, list):
                    current_neighbor_values = set(current_neighbors)
                else:
                    current_neighbor_values = set()
                
                action_neighbor_values = set()
                if isinstance(action.local_neighbors, list):
                    for n in action.local_neighbors:
                        if isinstance(n, dict):
                            action_neighbor_values.update(n.values())
                        elif isinstance(n, str):
                            action_neighbor_values.add(n)
                elif isinstance(action.local_neighbors, dict):
                    action_neighbor_values = set(action.local_neighbors.values())
                
                if len(current_neighbor_values & action_neighbor_values) >= 1:
                    similarity_score += 2
                    
            except Exception:
                pass
            
            if action.goal == current_context.get('goal', ''):
                similarity_score += 1
            
            if similarity_score >= 3:
                relevant_actions.append(action)
        
        relevant_actions.sort(key=lambda x: x.iteration, reverse=True)
        return relevant_actions[:5]
    
    def save(self):
        """Save history to disk - DISABLED for performance."""
        pass
    
    def load(self):
        """Load history from disk."""
        if not os.path.exists(self.save_path):
            return
        
        with open(self.save_path, 'r') as f:
            data = json.load(f)
        
        self.position = data.get("position", self.position)
        self.success_count = data.get("success_count", 0)
        self.failure_count = data.get("failure_count", 0)
        self.personal_patterns = data.get("personal_patterns", {})
        
        actions_data = data.get("action_history", [])
        self.action_history = [ActionRecord(**action_dict) for action_dict in actions_data]


class MemoryManager:
    """
    Central memory management system that coordinates global memory and local histories.
    Provides unified interface for agents to access learning information.
    """
    
    def __init__(self, log_dir: str = "memory_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.global_memory = GlobalMemory(os.path.join(log_dir, "global_memory.json"))
        self.local_histories: Dict[int, LocalHistory] = {}
        
        self.current_iteration = -1
        self.current_actions: List[ActionRecord] = []
        
        self.MAX_AGENTS = 50
        self.MAX_CURRENT_ACTIONS = 200
        self.EMERGENCY_AGENTS = 20
    
    def start_iteration(self, iteration: int, goal: str):
        """Start tracking a new iteration with aggressive cleanup."""
        self.current_iteration = iteration
        self.current_actions = []
        self.global_memory.update_goal(goal)
        
        self._aggressive_cleanup()
        
        if iteration % 2 == 0:
            gc.collect()
    
    def _aggressive_cleanup(self):
        """Extremely aggressive memory cleanup with emergency thresholds."""
        if len(self.local_histories) > self.MAX_AGENTS * 2:
            print(f"⚠️ EMERGENCY: Too many agents ({len(self.local_histories)}), keeping only {self.EMERGENCY_AGENTS}!")
            recent_agents = sorted(
                self.local_histories.items(),
                key=lambda x: len(x[1].action_history) + x[1].success_count + x[1].failure_count,
                reverse=True
            )[:self.EMERGENCY_AGENTS]
            self.local_histories = dict(recent_agents)
            gc.collect()
            
        elif len(self.local_histories) > self.MAX_AGENTS:
            recent_agents = sorted(
                self.local_histories.items(),
                key=lambda x: len(x[1].action_history),
                reverse=True
            )[:self.MAX_AGENTS]
            self.local_histories = dict(recent_agents)
        
        if len(self.current_actions) > self.MAX_CURRENT_ACTIONS:
            print(f"⚠️ Current actions buffer too large ({len(self.current_actions)}), emergency clear!")
            self.current_actions = self.current_actions[-self.MAX_CURRENT_ACTIONS//2:]
    
    def get_agent_memory_context(self, position: int, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive memory context for an agent making a decision.
        This includes both global insights and personal history with enhanced specificity.
        """
        if position not in self.local_histories:
            self.local_histories[position] = LocalHistory(position, os.path.join(self.log_dir, "local_histories"))
        
        local_history = self.local_histories[position]
        current_state = current_context.get('state', '')
        current_sequence = current_context.get('sequence', '')
        
        global_insights = self.global_memory.get_global_insights()
        personal_insights = local_history.get_personal_insights()
        relevant_experience = local_history.get_relevant_experience(current_context)
        position_specific_history = self._get_position_specific_history(position, current_state)
        recent_iteration_details = self._get_recent_iteration_details()
        neighboring_effects = self._get_neighboring_position_effects(position, current_sequence)
        sequence_context_patterns = self._get_sequence_context_patterns(position, current_sequence)
        energy_impact_analysis = self._get_energy_impact_analysis(current_state)
        mutation_recommendations = self._analyze_potential_mutations(current_state, global_insights, personal_insights)
        
        global_insights["current_iteration"] = self.current_iteration
        
        memory_context = {
            "global_insights": global_insights,
            "personal_insights": personal_insights,
            "relevant_experience": [asdict(action) for action in relevant_experience],
            "mutation_recommendations": mutation_recommendations,
            "learning_summary": self._generate_learning_summary(global_insights, personal_insights, current_state),
            "position_specific_history": position_specific_history,
            "recent_iteration_details": recent_iteration_details,
            "neighboring_effects": neighboring_effects,
            "sequence_context_patterns": sequence_context_patterns,
            "energy_impact_analysis": energy_impact_analysis
        }
        
        return memory_context
    
    def _get_position_specific_history(self, position: int, current_state: str) -> Dict[str, Any]:
        """Get detailed history of what happened at this exact position."""
        history = {"past_residues": [], "acceptance_pattern": [], "recent_changes": []}
        
        if position in self.local_histories:
            local_history = self.local_histories[position]
            for action in local_history.action_history[-10:]:
                if action.old_state != action.new_state:
                    reason = None
                    if action.was_accepted is not None:
                        matching_outcome = next(
                            (outcome for outcome in self.global_memory.iteration_outcomes 
                             if outcome.iteration == action.iteration and outcome.accepted == action.was_accepted),
                            None
                        )
                        if matching_outcome and matching_outcome.reason:
                            reason = matching_outcome.reason
                        else:
                            reason = "unknown reason" if action.was_accepted is False else "accepted"
                    
                    history["past_residues"].append({
                        "iteration": action.iteration,
                        "from": action.old_state,
                        "to": action.new_state,
                        "accepted": action.was_accepted,
                        "energy_change": action.energy_change,
                        "reason": reason,
                        "acceptance_status": "accepted" if action.was_accepted else "rejected"
                    })
        
        if history["past_residues"]:
            accepted_count = sum(1 for change in history["past_residues"] if change["accepted"])
            total_count = len(history["past_residues"])
            history["acceptance_rate"] = accepted_count / total_count if total_count > 0 else 0
            
            successful_residues = {}
            rejected_residues = {}
            for change in history["past_residues"]:
                if change["accepted"]:
                    to_residue = change["to"]
                    successful_residues[to_residue] = successful_residues.get(to_residue, 0) + 1
                else:
                    to_residue = change["to"]
                    rejected_residues[to_residue] = rejected_residues.get(to_residue, 0) + 1
            
            history["successful_residues"] = successful_residues
            history["rejected_residues"] = rejected_residues
            
            current_state_performance = [c for c in history["past_residues"] if c["to"] == current_state]
            if current_state_performance:
                history["current_state_performance"] = {
                    "times_tried": len(current_state_performance),
                    "times_accepted": sum(1 for c in current_state_performance if c["accepted"]),
                    "avg_energy_change": np.mean([c["energy_change"] for c in current_state_performance if c["energy_change"] is not None]) if any(c["energy_change"] is not None for c in current_state_performance) else None
                }
        
        return history
    
    def _get_recent_iteration_details(self) -> Dict[str, Any]:
        """Get comprehensive details about ALL iterations with full history."""
        details = {"full_iteration_history": [], "acceptance_patterns": {}, "energy_progression": []}
        
        all_outcomes = self.global_memory.iteration_outcomes
        
        for outcome in all_outcomes:
            iteration_detail = {
                "iteration": outcome.iteration,
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "energy": outcome.energy,
                "structure_score": outcome.structure_score,
                "mutation_count": outcome.mutation_count,
                "mutations_made": [],
                "energy_breakdown": {}
            }
            
            iteration_detail["energy_breakdown"] = {
                "total_energy": outcome.energy,
                "mutation_count": outcome.mutation_count
            }
            
            details["full_iteration_history"].append(iteration_detail)
        
        if all_outcomes:
            details["acceptance_patterns"] = {
                "total_iterations": len(all_outcomes),
                "total_accepted": sum(1 for o in all_outcomes if o.accepted),
                "overall_acceptance_rate": sum(1 for o in all_outcomes if o.accepted) / len(all_outcomes),
                "recent_5_acceptance_rate": sum(1 for o in all_outcomes[-5:] if o.accepted) / min(5, len(all_outcomes)),
                "acceptance_trend": self._analyze_acceptance_trend(all_outcomes),
                "rejection_reasons": self._analyze_rejection_reasons(all_outcomes)
            }
            
            details["energy_progression"] = self._analyze_energy_progression(all_outcomes)
        
        return details
    
    def _get_neighboring_position_effects(self, position: int, current_sequence: str) -> Dict[str, Any]:
        """Analyze neighboring sequences (accepted/rejected) and their effects with energy data."""
        effects = {
            "neighboring_accepted_sequences": [],
            "neighboring_rejected_sequences": [],
            "sequence_similarity_analysis": {},
            "local_mutation_effects": {},
            "sequence_context": ""
        }
        
        current_context = self._get_local_sequence_context(current_sequence, position, 5)
        effects["sequence_context"] = current_context
        
        for outcome in self.global_memory.iteration_outcomes:
            sequence_info = {
                "iteration": outcome.iteration,
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "energy": outcome.energy,
                "structure_score": outcome.structure_score,
                "mutation_count": outcome.mutation_count,
                "sequence": getattr(outcome, 'sequence', 'N/A'),
                "mutations_in_vicinity": []
            }
            
            if outcome.accepted:
                effects["neighboring_accepted_sequences"].append(sequence_info)
            else:
                effects["neighboring_rejected_sequences"].append(sequence_info)
        
        effects["neighboring_rejected_sequences"].sort(key=lambda x: x["energy"], reverse=True)
        
        pareto_optimal_accepted = self._get_pareto_optimal_sequences(effects["neighboring_accepted_sequences"])
        
        effects["local_mutation_effects"] = self._analyze_local_mutation_patterns(
            effects["neighboring_accepted_sequences"] + effects["neighboring_rejected_sequences"],
            position
        )
        
        effects["sequence_similarity_analysis"] = {
            "best_accepted": pareto_optimal_accepted[:5],
            "worst_rejected": effects["neighboring_rejected_sequences"][:3]
        }
        
        return effects
    
    def _get_pareto_optimal_sequences(self, accepted_sequences: List[Dict]) -> List[Dict]:
        """
        Get Pareto optimal sequences using the criteria:
        1. Higher score is always better (prioritize structure score)
        2. Same score: lower energy is better  
        3. Lower score: reject
        """
        if not accepted_sequences:
            return []
        
        sorted_sequences = sorted(accepted_sequences, key=lambda x: (-x["structure_score"], x["energy"]))
        
        pareto_optimal = []
        
        for seq in sorted_sequences:
            is_dominated = False
            current_score = seq["structure_score"]
            current_energy = seq["energy"]
            
            for optimal_seq in pareto_optimal:
                optimal_score = optimal_seq["structure_score"]
                optimal_energy = optimal_seq["energy"]
                
                if (optimal_score > current_score or 
                    (optimal_score == current_score and optimal_energy < current_energy)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal = [
                    opt_seq for opt_seq in pareto_optimal
                    if not (current_score > opt_seq["structure_score"] or 
                           (current_score == opt_seq["structure_score"] and current_energy < opt_seq["energy"]))
                ]
                pareto_optimal.append(seq)
        
        pareto_optimal.sort(key=lambda x: (-x["structure_score"], x["energy"]))
        
        return pareto_optimal
    
    def _get_local_sequence_context(self, sequence: str, position: int, window: int) -> str:
        """Get local sequence context around a position."""
        start = max(0, position - window)
        end = min(len(sequence), position + window + 1)
        return sequence[start:end]
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate overall sequence similarity."""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        if matches == 0:
            return 0.0
        else:
            return matches / len(seq1)
    
    def _calculate_local_context_similarity(self, context1: str, context2: str) -> float:
        """Calculate similarity between local contexts."""
        if len(context1) != len(context2):
            min_len = min(len(context1), len(context2))
            context1 = context1[:min_len]
            context2 = context2[:min_len]
        
        if not context1:
            return 0.0
            
        matches = sum(1 for a, b in zip(context1, context2) if a == b)
        return matches / len(context1)
    
    def _analyze_acceptance_trend(self, outcomes: List) -> str:
        """Analyze the trend in acceptance rates over iterations."""
        if len(outcomes) < 2:
            return "fewer than 2 iterations"
        
        mid_point = len(outcomes) // 2
        first_half_rate = sum(1 for o in outcomes[:mid_point] if o.accepted) / mid_point
        second_half_rate = sum(1 for o in outcomes[mid_point:] if o.accepted) / (len(outcomes) - mid_point)
        
        if second_half_rate > first_half_rate + 0.2:
            return "improving"
        elif second_half_rate < first_half_rate - 0.2:
            return "declining"
        else:
            return "stable"
    
    def _analyze_rejection_reasons(self, outcomes: List) -> Dict[str, int]:
        """Analyze common reasons for rejection."""
        reasons = {}
        for outcome in outcomes:
            if not outcome.accepted:
                reason = outcome.reason.lower() if outcome.reason else "unknown"
                if "energy" in reason:
                    category = "energy is too high, consider protein folding principles"
                elif "structure" in reason or "score" in reason:
                    category = "design goal is not satisfied, improve the structure score"
                elif "diversity" in reason or "repetitive" in reason:
                    category = "lack of diversity, try bold mutations"
                else:
                    category = "refer to the decision rules and folding principles"
                
                reasons[category] = reasons.get(category, 0) + 1
        
        return reasons
    
    def _analyze_energy_progression(self, outcomes: List) -> Dict[str, Any]:
        """Analyze how energy has changed over iterations."""
        progression = {
            "energy_history": [],
            "best_energy": float('inf'),
            "worst_energy": float('-inf'),
            "energy_trend": "unknown",
            "recent_energy_change": 0
        }
        
        for outcome in outcomes:
            if hasattr(outcome, 'energy_data') and outcome.energy_data and 'total_energy' in outcome.energy_data:
                energy = outcome.energy_data['total_energy']
                energy_entry = {
                    "iteration": outcome.iteration,
                    "energy": energy,
                    "accepted": outcome.accepted
                }
                
                # For rejected iterations, also store the proposed energy if available
                if not outcome.accepted and 'proposed_energy' in outcome.energy_data:
                    energy_entry["proposed_energy"] = outcome.energy_data['proposed_energy']
                    energy_entry["proposed_energy_change"] = outcome.energy_data.get('proposed_energy_change', 0)
                
                progression["energy_history"].append(energy_entry)
                
                if energy < progression["best_energy"]:
                    progression["best_energy"] = energy
                if energy > progression["worst_energy"]:
                    progression["worst_energy"] = energy
        
        # Analyze trend
        if len(progression["energy_history"]) >= 3:
            recent_energies = [e["energy"] for e in progression["energy_history"][-3:]]
            early_energies = [e["energy"] for e in progression["energy_history"][:3]]
            
            recent_avg = sum(recent_energies) / len(recent_energies)
            early_avg = sum(early_energies) / len(early_energies)
            
            progression["recent_energy_change"] = recent_avg - early_avg
            
            if progression["recent_energy_change"] < -1.0:
                progression["energy_trend"] = "improving"
            elif progression["recent_energy_change"] > 1.0:
                progression["energy_trend"] = "worsening"
            else:
                progression["energy_trend"] = "stable"
        
        # Analyze structure score vs energy relationships
        progression["structure_energy_analysis"] = self._analyze_structure_energy_relationships(outcomes)
        
        return progression
    
    def _analyze_structure_energy_relationships(self, outcomes: List) -> Dict[str, Any]:
        """Analyze the relationship between structure scores and energies."""
        analysis = {
            "score_energy_pairs": [],
            "best_energy_by_score": {},
            "pareto_efficient_points": [],
            "current_vs_best_same_score": {}
        }
        
        # Collect structure score and energy pairs
        for outcome in outcomes:
            if (hasattr(outcome, 'energy_data') and outcome.energy_data and 
                'total_energy' in outcome.energy_data and outcome.energy_data['total_energy'] != float('inf')):
                
                score = outcome.structure_score
                energy = outcome.energy_data['total_energy']
                
                analysis["score_energy_pairs"].append({
                    "score": score,
                    "energy": energy,
                    "iteration": outcome.iteration,
                    "accepted": outcome.accepted
                })
                
                # Track best energy for each score (rounded to nearest 0.5)
                score_bucket = round(score * 2) / 2  # Round to nearest 0.5
                if score_bucket not in analysis["best_energy_by_score"]:
                    analysis["best_energy_by_score"][score_bucket] = {
                        "energy": energy,
                        "iteration": outcome.iteration,
                        "accepted": outcome.accepted
                    }
                elif energy < analysis["best_energy_by_score"][score_bucket]["energy"]:
                    analysis["best_energy_by_score"][score_bucket] = {
                        "energy": energy,
                        "iteration": outcome.iteration,
                        "accepted": outcome.accepted
                    }
        
        # Find Pareto efficient points (best energy for each structure score level)
        if analysis["score_energy_pairs"]:
            # Sort by structure score descending, then by energy ascending
            sorted_pairs = sorted(analysis["score_energy_pairs"], key=lambda x: (-x["score"], x["energy"]))
            
            pareto_points = []
            min_energy_so_far = float('inf')
            
            for point in sorted_pairs:
                if point["energy"] < min_energy_so_far:
                    pareto_points.append(point)
                    min_energy_so_far = point["energy"]
            
            analysis["pareto_efficient_points"] = pareto_points
            
            # Analyze current iteration vs best same score
            if outcomes:
                latest = outcomes[-1]
                latest_score = latest.structure_score
                latest_energy = latest.energy_data.get('total_energy', float('inf')) if hasattr(latest, 'energy_data') and latest.energy_data else float('inf')
                
                # Find best energy with same score (±0.5 tolerance)
                same_score_energies = [
                    p["energy"] for p in analysis["score_energy_pairs"] 
                    if abs(p["score"] - latest_score) <= 0.5 and p["energy"] != float('inf')
                ]
                
                if same_score_energies:
                    best_same_score_energy = min(same_score_energies)
                    energy_gap = latest_energy - best_same_score_energy
                    
                    analysis["current_vs_best_same_score"] = {
                        "current_score": latest_score,
                        "current_energy": latest_energy,
                        "best_energy_same_score": best_same_score_energy,
                        "energy_gap": energy_gap,
                        "has_improvement_potential": energy_gap > 1.0
                    }
        
        return analysis
    
    def _analyze_local_mutation_patterns(self, all_sequences: List, position: int) -> Dict[str, Any]:
        """Analyze patterns in mutations near the current position."""
        patterns = {
            "successful_nearby_mutations": {},
            "failed_nearby_mutations": {},
            "position_specific_preferences": {},
            "distance_effect_analysis": {}
        }
        
        for seq_info in all_sequences:
            for mutation in seq_info["mutations_in_vicinity"]:
                mut_pos = mutation["position"]
                distance = abs(mut_pos - position)
                change = mutation["change"]
                
                # Track by acceptance
                if seq_info["accepted"]:
                    patterns["successful_nearby_mutations"][change] = patterns["successful_nearby_mutations"].get(change, 0) + 1
                    
                    # Position-specific tracking
                    if mut_pos not in patterns["position_specific_preferences"]:
                        patterns["position_specific_preferences"][mut_pos] = {"successful": [], "failed": []}
                    patterns["position_specific_preferences"][mut_pos]["successful"].append(change)
                else:
                    patterns["failed_nearby_mutations"][change] = patterns["failed_nearby_mutations"].get(change, 0) + 1
                    
                    # Position-specific tracking
                    if mut_pos not in patterns["position_specific_preferences"]:
                        patterns["position_specific_preferences"][mut_pos] = {"successful": [], "failed": []}
                    patterns["position_specific_preferences"][mut_pos]["failed"].append(change)
                
                # Distance effect analysis
                if distance not in patterns["distance_effect_analysis"]:
                    patterns["distance_effect_analysis"][distance] = {"successful": 0, "failed": 0}
                
                if seq_info["accepted"]:
                    patterns["distance_effect_analysis"][distance]["successful"] += 1
                else:
                    patterns["distance_effect_analysis"][distance]["failed"] += 1
        
        return patterns
    
    def _get_sequence_context_patterns(self, position: int, current_sequence: str) -> Dict[str, Any]:
        """Find successful patterns in similar sequence contexts."""
        patterns = {"similar_contexts": [], "successful_motifs": []}
        
        seq_length = len(current_sequence)
        context_start = max(0, position - 2)
        context_end = min(seq_length, position + 3)
        current_context = current_sequence[context_start:context_end]
        
        for outcome in self.global_memory.iteration_outcomes[-3:]:
            if outcome.accepted:
                patterns["similar_contexts"].append({
                    "iteration": outcome.iteration,
                    "energy": outcome.energy,
                    "structure_score": outcome.structure_score,
                    "mutation_count": outcome.mutation_count
                })
        
        patterns["similar_contexts"].sort(key=lambda x: x["energy"])
        
        return patterns
    
    def _get_energy_impact_analysis(self, current_state: str) -> Dict[str, Any]:
        """Analyze energy impacts of different residue types."""
        analysis = {"energy_improving_changes": [], "energy_worsening_changes": [], "neutral_changes": []}
        
        for outcome in self.global_memory.iteration_outcomes:
            change_info = {
                "iteration": outcome.iteration,
                "energy": outcome.energy,
                "accepted": outcome.accepted,
                "mutation_count": outcome.mutation_count
            }
            
            if outcome.energy < -100:
                analysis["energy_improving_changes"].append(change_info)
            elif outcome.energy > 100:
                analysis["energy_worsening_changes"].append(change_info)
            else:
                analysis["neutral_changes"].append(change_info)
        
        for category in analysis:
            analysis[category].sort(key=lambda x: x["energy"])
        
        return analysis
    
    def _analyze_potential_mutations(self, current_state: str, global_insights: Dict, personal_insights: Dict) -> Dict[str, Any]:
        """Analyze potential mutations based on global and local patterns using cumulative data."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        recommendations = {"highly_recommended": [], "recommended": [], "avoid": []}
        
        for aa in amino_acids:
            if aa == current_state:
                continue
            
            pattern = f"{current_state}->{aa}"
            
            global_rate = self.global_memory.get_pattern_success_rate(pattern)
            
            personal_rate = personal_insights.get("preferred_mutations", {})
            personal_rate_value = next((rate for p, rate in personal_rate if p == pattern), 0.5)
            
            combined_score = (global_rate * 0.7 + personal_rate_value * 0.3)
            
            pattern_data = None
            if hasattr(self.global_memory, 'cumulative_patterns') and pattern in self.global_memory.cumulative_patterns:
                successes, total = self.global_memory.cumulative_patterns[pattern]
                pattern_data = f"{global_rate:.2f} ({successes}/{total})"
            else:
                pattern_data = f"{global_rate:.2f}"
            
            if combined_score > 0.7:
                recommendations["highly_recommended"].append((aa, combined_score, pattern_data))
            elif combined_score > 0.5:
                recommendations["recommended"].append((aa, combined_score, pattern_data))
            elif combined_score < 0.3:
                recommendations["avoid"].append((aa, combined_score, pattern_data))
        
        for category in recommendations:
            recommendations[category].sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def _generate_learning_summary(self, global_insights: Dict, personal_insights: Dict, current_state: str) -> str:
        """Generate highly specific, actionable learning summary for the agent."""
        insights = []
        
        if hasattr(self, 'current_iteration') and self.current_iteration >= 0:
            recent_outcomes = self.global_memory.iteration_outcomes[-3:] if self.global_memory.iteration_outcomes else []
            
            if recent_outcomes:
                recent_details = self._analyze_recent_specific_patterns(recent_outcomes)
                insights.extend(self._generate_specific_insights(recent_details, global_insights, personal_insights, current_state))
        
        if not insights:
            insights = self._generate_enhanced_fallback_insights(global_insights, personal_insights, current_state)
        
        return " ".join(insights[:3])
    
    def _analyze_recent_specific_patterns(self, recent_outcomes: List) -> Dict[str, Any]:
        """Analyze specific patterns in recent outcomes."""
        analysis = {
            "energy_patterns": {},
            "mutation_patterns": {},
            "position_patterns": {},
            "rejection_specifics": []
        }
        
        for outcome in recent_outcomes:
            if hasattr(outcome, 'energy_data') and outcome.energy_data:
                total_energy = outcome.energy_data.get('total_energy', 0)
                analysis["energy_patterns"][outcome.iteration] = {
                    "energy": total_energy,
                    "accepted": outcome.accepted,
                    "mutations_count": len([a for a in outcome.agent_actions if a.old_state != a.new_state])
                }
            
            if outcome.mutation_count > 0:
                analysis["mutation_patterns"][f"iter_{outcome.iteration}"] = {
                    "attempts": outcome.mutation_count,
                    "successes": 1 if outcome.accepted else 0,
                    "energy": outcome.energy
                }
            
            if not outcome.accepted:
                analysis["rejection_specifics"].append({
                    "iteration": outcome.iteration,
                    "reason": outcome.reason,
                    "mutation_count": outcome.mutation_count,
                    "energy": outcome.energy
                })
        
        return analysis
    
    def _generate_specific_insights(self, recent_details: Dict, global_insights: Dict, personal_insights: Dict, current_state: str) -> List[str]:
        """Generate specific, actionable insights from detailed analysis."""
        insights = []
        
        energy_patterns = recent_details.get("energy_patterns", {})
        if energy_patterns:
            energies = [ep["energy"] for ep in energy_patterns.values() if ep["energy"] != 0]
            if energies:
                best_energy = min(energies)
                recent_avg_energy = None
                if len(energies) > 5:
                    recent_energies = energies[-5:]
                    recent_avg_energy = sum(recent_energies) / len(recent_energies)
                
                if energies[-1] - best_energy > 400:
                    insights.append(f"This iteration's energy is {energies[-1] - best_energy:.1f} units higher than the best energy so far, consider energy-lowering mutations")
                
                if recent_avg_energy is not None:
                    if recent_avg_energy > 400:
                        insights.append(f"Recent 5 iterations' average energy is high ({recent_avg_energy:.1f}) - prioritize energy-lowering mutations")
                    else:
                        insights.append(f"Energy is stable (recent 5 iterations' average energy: {recent_avg_energy:.1f}) - consider design goal-improving mutations")
        
        mutation_patterns = recent_details.get("mutation_patterns", {})
        if mutation_patterns:
            successful_mutations = {mut: data["successes"]/data["attempts"] 
                                  for mut, data in mutation_patterns.items() 
                                  if data["attempts"] > 0}
            
            if successful_mutations:
                best_mutation = max(successful_mutations.items(), key=lambda x: x[1])
                worst_mutations = [mut for mut, rate in successful_mutations.items() if rate == 0]
                
                if best_mutation[1] > 0.5:
                    insights.append(f"\nMutation {best_mutation[0]} has {best_mutation[1]:.0%} success rate - similar hydrophobic/polar swaps recommended")
                
                if worst_mutations and current_state in [m.split('->')[0] for m in worst_mutations]:
                    failed_targets = [m.split('->')[1] for m in worst_mutations if m.startswith(current_state)]
                    insights.append(f"Avoid {current_state}->{failed_targets} - these failed recently")
        
        rejection_specifics = recent_details.get("rejection_specifics", [])
        if rejection_specifics:
            energy_rejections = [r for r in rejection_specifics if r["reason"] and "energy" in r["reason"].lower()]
            structure_rejections = [r for r in rejection_specifics if r["reason"] and "structure" in r["reason"].lower()]
            
            if len(energy_rejections) >= 2:
                insights.append("Multiple energy rejections - use energy-favorable mutations (A->V, S->A, E->A)")
            
            if len(structure_rejections) >= 2:
                insights.append("Structure quality issues - avoid large sidechains in constrained regions")
        
        return insights
    
    def _generate_enhanced_fallback_insights(self, global_insights: Dict, personal_insights: Dict, current_state: str) -> List[str]:
        """Generate enhanced but simpler insights when detailed analysis isn't available."""
        insights = []
        
        acceptance_rate = global_insights.get("recent_acceptance_rate", 0)
        energy_trend = global_insights.get("energy_trend", "unknown")
        
        if acceptance_rate < 0.3 and energy_trend == "worsening":
            insights.append(f"Low acceptance ({acceptance_rate:.0%}) + rising energy - use energy-stabilizing mutations (hydrophobic core: A,V,I,L)")
        elif acceptance_rate < 0.3:
            insights.append(f"Low acceptance ({acceptance_rate:.0%}) - try bolder mutations: similar size/polarity substitutions")
        elif acceptance_rate > 0.7:
            insights.append(f"High acceptance ({acceptance_rate:.0%}) - current strategy works, fine-tune with bolder mutations")
        
        personal_rate = personal_insights.get("overall_success_rate", 0)
        recent_trend = personal_insights.get("recent_trend", "unknown")
        
        if personal_rate < 0.4 and recent_trend == "declining":
            insights.append(f"Your performance declining ({personal_rate:.0%}) - copy successful patterns from high-performing agents")
        elif personal_rate > 0.6:
            insights.append(f"You're performing well ({personal_rate:.0%}) - maintain current mutation strategy")
        
        best_patterns = global_insights.get("best_patterns", [])
        if best_patterns:
            best_pattern = best_patterns[0]
            if best_pattern[0].split('->')[0] == current_state:
                target = best_pattern[0].split('->')[1]
                insights.append(f"From {current_state}: {target} has {best_pattern[1]:.0%} success rate - highly recommended")
            else:
                insights.append(f"Top pattern {best_pattern[0]} ({best_pattern[1]:.0%}) - apply similar chemical logic")
        
        return insights
    
    def record_agent_action(self, position: int, old_state: str, new_state: str, context: Dict[str, Any]):
        """Record an agent's action with emergency cleanup."""
        action = ActionRecord(
            iteration=self.current_iteration,
            position=position,
            old_state=old_state,
            new_state=new_state,
            goal=context.get('goal', '')
        )
        
        self.current_actions.append(action)
        
        if len(self.current_actions) > self.MAX_CURRENT_ACTIONS:
            print(f"⚠️ Emergency cleanup: current_actions buffer too large ({len(self.current_actions)})")
            self.current_actions = self.current_actions[-self.MAX_CURRENT_ACTIONS//2:]
            gc.collect()
    
    def finalize_iteration(self, initial_sequence: str, proposed_sequence: str, final_sequence: str, 
                          accepted: bool, reason: str, energy_data: Dict, structure_score: float):
        """Finalize iteration with aggressive memory management."""
        mutation_count = len([a for a in self.current_actions if a.old_state != a.new_state])
        
        for action in self.current_actions:
            action.was_accepted = accepted
            action.energy_change = energy_data.get('total_energy', 0) if accepted else None
            
            if action.old_state != action.new_state:
                pattern = f"{action.old_state}->{action.new_state}"
                self.global_memory.add_pattern_outcome(pattern, accepted)
        
        outcome = IterationOutcome(
            iteration=self.current_iteration,
            accepted=accepted,
            reason=reason[:50] if reason else "",
            energy=energy_data.get('total_energy', 0),
            structure_score=structure_score,
            mutation_count=mutation_count,
            sequence=final_sequence
        )
        
        self.global_memory.add_iteration_outcome(outcome)
        
        active_positions = set(a.position for a in self.current_actions if a.old_state != a.new_state)
        for position in active_positions:
            if position not in self.local_histories:
                self.local_histories[position] = LocalHistory(position, os.path.join(self.log_dir, "local_histories"))
            
            position_actions = [a for a in self.current_actions if a.position == position]
            for action in position_actions:
                self.local_histories[position].add_action(action)
        
        self.current_actions = []
        
        self._aggressive_cleanup()
        gc.collect()
    

    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics with current limits."""
        return {
            "global_memory": {
                "iterations": len(self.global_memory.iteration_outcomes),
                "max_iterations": self.global_memory.MAX_ITERATIONS,
                "patterns": len(self.global_memory.successful_patterns) + len(self.global_memory.failed_patterns),
                "max_patterns": self.global_memory.MAX_PATTERNS * 2
            },
            "local_histories": {
                "tracked_agents": len(self.local_histories),
                "max_agents": self.MAX_AGENTS,
                "total_actions": sum(len(history.action_history) for history in self.local_histories.values()),
                "current_actions_buffer": len(self.current_actions),
                "max_current_actions": self.MAX_CURRENT_ACTIONS
            }
        } 