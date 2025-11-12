"""LLM interface for the agents."""

from typing import Dict
from openai import OpenAI
from local_config import API_KEY
from constants import MODEL, DEFAULT_NEIGHBORS, DEFAULT_CUTOFF
from config import DesignProposal

_IN_AGENT_PROCESSING = False

client = OpenAI(api_key=API_KEY)


def call_llm(
    role: str,
    query: str,
    messages: list = None,
    system_message: str = "You are a helpful assistant, who uses a memory of information to make decisions.",
    figure_base64: str = None,
    response_format=None,
    history=None
):
    """
    Call the LLM with the given parameters.
    
    Args:
        role: Role of the agent
        query: Query to send to the LLM
        messages: Optional pre-built messages
        system_message: System message for the LLM
        figure_base64: Optional base64 encoded figure
        response_format: Optional structured response format
        history: Optional conversation history
    
    Returns:
        Response from the LLM (structured or text)
    """
    if messages is None:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
    
    if history:
        formatted_history = []
        for msg in history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                formatted_history.append(msg)
            else:
                print(f"Warning: Skipping malformed history message: {msg}")
        
        if formatted_history:
            messages = messages[:-1] + formatted_history + [messages[-1]]
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            print(f"Error: Invalid message format at index {i}: {msg}")
            if not isinstance(msg, dict):
                messages[i] = {"role": "user", "content": str(msg)}
            elif 'role' not in msg:
                messages[i]['role'] = "user"
            elif 'content' not in msg:
                messages[i]['content'] = ""
    
    if figure_base64:
        if messages[-1]["role"] == "user":
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{figure_base64}"}}
            ]
    
    if response_format:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            response_format=response_format
        )
        
        if response.choices[0].message.parsed:
            return response.choices[0].message.parsed.model_dump()
        else:
            return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return response.choices[0].message.content


def call_llm_agent(agent_input: Dict, config, history=None, current_energies=None, 
                   residue_energy=None, rejected_sequences=None, structure_feedback=None,
                   distance_matrix_info=None) -> str:
    """
    Generate mutation suggestions for a given position using the LLM agent with memory enhancement.
    During iteration, this should NOT use any structure-dependent information.
    
    Args:
        agent_input: Dictionary with position, state, sequence, neighbors, goal, and memory_context
        config: Design configuration
        history: Previous iteration history (unused during iteration) 
        current_energies: Energy data (unused during iteration)
        residue_energy: Per-residue energy (unused during iteration) 
        rejected_sequences: List of previously rejected sequences
        structure_feedback: Structure evaluation from previous iteration
        distance_matrix_info: Distance matrix analysis (unused during iteration)
        
    Returns:
        Single character amino acid code
    """
    global _IN_AGENT_PROCESSING
    
    _IN_AGENT_PROCESSING = True
    
    try:
        prompt_parts = []
        
        if hasattr(config, 'get_goal_name') and hasattr(config, 'get_goal_description'):
            goal_name = config.get_goal_name()
            goal_description = config.get_goal_description()
        else:
            goal_input = agent_input['goal']
            if isinstance(goal_input, dict):
                goal_name = goal_input.get('name', 'unknown')
                goal_description = goal_input.get('description', goal_input.get('name', 'unknown'))
            else:
                goal_name = str(goal_input)
                goal_description = str(goal_input)

        prompt_parts.append("PART 1: Your Role and Task\n")
        
        prompt_parts.append(f"Design Goal: {goal_name}")
        if goal_description is not None and goal_description != goal_name:
            prompt_parts.append(f"Design Goal Description: {goal_description}")
        prompt_parts.append(f"Position: {agent_input['position'] + 1}")
        prompt_parts.append(f"Current residue: {agent_input['state']}")
        prompt_parts.append(f"Current full sequence: {agent_input['sequence']}")
        
        prompt_parts.append(f"""
Decision Rules:
1. Most importantly, consider the design goal: {goal_description}
2. Learn from the memory history, context of the current position, sequence and spatial neighbors, and structure and energy feedback to inform decision on no mutation or mutation
3. Consider fundamental folding principles
4. Maintain sequence diversity to avoid repetitive residues but still consider residues mentioned in the design goal

Fundamental Folding Principles:
- Minimize disruption to secondary structures
- Favor compact, stable folding with low Rosetta energy
- Prefer conservative mutations unless necessary
- Consider hydrophobic core stability and surface accessibility

Your Task:
Choose the best amino acid for position {agent_input['position'] + 1} to achieve the design goal: {goal_description} considering the above decision rules and fundamental folding principles""")
        
        prompt_parts.append("\nPART 2: Local Neighborhood Context")
        secondary_structure = agent_input.get('secondary_structure', None)
        if secondary_structure:
            prompt_parts.append(f"Secondary structure:")
            dssp_assignment = secondary_structure.get('dssp_assignment', 'unknown')
            dssp_available = secondary_structure.get('dssp_available', False)
            
            dssp_mapping = {
                'helix': 'Alpha helix (H/G/I)',
                'sheet': 'Beta sheet/strand (E/B)', 
                'turn': 'Turn/bend/coil (T/S/-)',
                'loop': 'Loop/coil',
                'unknown': 'Unknown/unavailable'
            }
            
            dssp_description = dssp_mapping.get(dssp_assignment, dssp_assignment)
            prompt_parts.append(f"- Position {agent_input['position'] + 1} secondary structure: {dssp_description}")
            prompt_parts.append(f"- Analysis method: {'DSSP (high confidence)' if dssp_available else 'Distance-based fallback'}")
            
            if dssp_assignment == 'helix':
                prompt_parts.append("- Helix (3, 4, or 5-turn helix)")
            elif dssp_assignment == 'sheet':
                prompt_parts.append("- Beta sheet (parallel or anti-parallel) or strand")
            elif dssp_assignment in ['turn', 'loop']:
                prompt_parts.append("- Turn/bend/coil/loop")
        else:
            prompt_parts.append(f"\nSecondary structure: No DSSP analysis available")
        
        neighbors = agent_input['neighbors']
        if neighbors:
            neighbor_info = []
            if 'n-term' in neighbors:
                neighbor_info.append(f"{DEFAULT_NEIGHBORS} N-terminus neighbor(s): {neighbors['n-term']}")
            if 'c-term' in neighbors:
                neighbor_info.append(f"{DEFAULT_NEIGHBORS} C-terminus neighbor(s): {neighbors['c-term']}")
            if 'linear' in neighbors:
                neighbor_info.append(f"Linear neighbors: {neighbors['linear']}")
            if neighbor_info:
                prompt_parts.append(f"\nLinear neighbors: {', '.join(neighbor_info)}")
        
        spatial_neighbors = agent_input['spatial_neighbors']
        if spatial_neighbors and len(spatial_neighbors) > 0:
            prompt_parts.append(f"\nSpatial neighbors from distance matrix within {DEFAULT_CUTOFF} Å: {len(spatial_neighbors)} total")
            
            spatial_details = []
            for i, neighbor in enumerate(spatial_neighbors): 
                pos = neighbor.get('position', '?') + 1
                state = neighbor.get('state', '?')
                distance = neighbor.get('distance', None)
                if distance is not None:
                    spatial_details.append(f"- Position {pos}: {state} (distance: {distance:.1f}Å)")
                else:
                    spatial_details.append(f"- Position {pos}: {state}")
            
            if spatial_details:
                prompt_parts.append(f"Detailed spatial neighbors sorted by distance:")
                prompt_parts.extend(spatial_details)
            
            exposure = agent_input['exposure']
            if exposure == "buried":
                prompt_parts.append(f"Spatial context: Highly connected position ({len(spatial_neighbors)} neighbors) - likely buried")
                prompt_parts.append("Mutate or keep the current residue according to the design goal; might favor hydrophobic, compact, and core-stabilizing residues.")

            elif exposure == "surface":
                prompt_parts.append(f"Spatial context: Not so spatially connected ({len(spatial_neighbors)} neighbors) - likely exposed surface")
                prompt_parts.append("Mutate or keep the current residue according to the design goal; might favor polar, charged, or functionally relevant residues.")
            
        else:
            prompt_parts.append(f"\nSpatial neighbors: None available ")
        
        structural_context = agent_input.get('structural_context', None)
        if structural_context:
            prompt_parts.append(f"\nDetailed structural context from distance matrix analysis:")
            prompt_parts.append(f"Structural summary:")
            prompt_parts.append(f"{structural_context['structural_summary']}")

        memory_context = agent_input.get('memory_context', {})
        if memory_context:
            memory_section = []
            
            personal_insights = memory_context.get('personal_insights', {})
            position_history = memory_context.get('position_specific_history', {})
            iteration_details = memory_context.get('recent_iteration_details', {})
            
            global_insights = memory_context.get('global_insights', {})

            if global_insights.get('total_iterations', 0) > 0:
                prompt_parts.append("\nPART 3: Memory History and Analysis\n")
            
            if global_insights and global_insights.get('total_iterations', 0) > 0:
                memory_section.append(f"Global Patterns from All Agents from Memory:")
                memory_section.append(f"- Total iterations completed: {global_insights.get('total_iterations', 0)}")
                memory_section.append(f"- Overall acceptance rate: {global_insights.get('acceptance_rate', 0):.1%}")
                if global_insights.get('total_iterations') > 5:
                    memory_section.append(f"- Recent acceptance rate in the last 5 iterations: {global_insights.get('recent_acceptance_rate', 0):.1%}")
                memory_section.append(f"- Energy trend: {global_insights.get('energy_trend', 'unknown')}")
                
                if global_insights.get('best_patterns'):
                    best_patterns = global_insights['best_patterns']
                    memory_section.append(f"- Most successful mutation patterns across all agents: {[f'{p[0]} ({p[1]:.1%})' for p in best_patterns]}")
                
                if global_insights.get('worst_patterns'):
                    worst_patterns = global_insights['worst_patterns']
                    memory_section.append(f"- Least likely to be successful mutation patterns across all agents: {[f'{p[0]} ({p[1]:.1%})' for p in worst_patterns]}")

                if iteration_details and iteration_details.get('full_iteration_history'):
                    history = iteration_details['full_iteration_history']
                    accepted_iters = [f"Iter{iter_detail['iteration']}({iter_detail['energy']:.1f})" for iter_detail in history if iter_detail['accepted']]
                    rejected_iters = [f"Iter{iter_detail['iteration']}({iter_detail['energy']:.1f})" for iter_detail in history if not iter_detail['accepted']]
                    
                    memory_section.append(f"\nPrevious iteration outcomes:")
                    if accepted_iters:
                        memory_section.append(f"- Accepted iterations and energies: {accepted_iters}")
                    if rejected_iters:
                        memory_section.append(f"- Rejected iterations and energies: {rejected_iters}")
            
            if position_history and position_history.get('past_residues'):
                memory_section.append(f"\nPersonal Mutations at Position {agent_input['position'] + 1} and Local Neighborhood Mutations from Memory:")
                
                position_changes = position_history['past_residues']
                full_history = iteration_details.get('full_iteration_history', []) if iteration_details else []
                
                iter_lookup = {iter_detail['iteration']: iter_detail for iter_detail in full_history}
                
                for change in reversed(position_changes):
                    iter_num = change['iteration']
                    status = "ACCEPTED" if change['accepted'] else "REJECTED"
                    mutation = f"{change['from']}->{change['to']}"
                    
                    reason = change.get('reason', 'unknown reason')
                    
                    memory_section.append(f"Iter {iter_num}: {mutation} {status} for reason of {reason}")
                    
                    iter_detail = iter_lookup.get(iter_num, {})
                    
                    if iter_detail:
                        memory_section.append(f"  Iteration details: Energy: {iter_detail.get('energy', 'N/A'):.1f}, Score: {iter_detail.get('structure_score', 'N/A'):.1f}, {iter_detail.get('mutation_count', 0)} mutations")
                    
                    if iter_detail and iter_detail.get('mutations_made'):
                        current_pos = agent_input['position']
                        nearby_mutations = []
                        for mut in iter_detail['mutations_made']:
                            if mut['position'] == current_pos:
                                continue
                            elif abs(mut['position'] - current_pos) <= DEFAULT_NEIGHBORS: 
                                nearby_mutations.append(f"position {mut['position']+1}: {mut['change']}")
                        if nearby_mutations:
                            memory_section.append(f"  Nearby mutations within ±{DEFAULT_NEIGHBORS} positions: {', '.join(nearby_mutations)}")
                    
                    if iter_detail:
                        total_energy = iter_detail.get('energy', 'N/A')
                        memory_section.append(f"  Total energy: {total_energy:.1f}")
                    elif change.get('energy_change') is not None:
                        memory_section.append(f"  Energy change: {change['energy_change']:.2f}")
                    else:
                        memory_section.append(f"  Total energy: N/A")
                    
                    memory_section.append("")
                
                neighboring_effects = memory_context.get('neighboring_effects', {})
                if neighboring_effects:
                    memory_section.append(f"\nLocal Neighborhood Mutations Analysis:")
                    memory_section.append(f"- Current local context: {neighboring_effects.get('sequence_context', '')}")
                    
                    best_accepted = neighboring_effects.get('sequence_similarity_analysis', {}).get('best_accepted', [])
                    if best_accepted:
                        memory_section.append(f"- Pareto optimal sequences (score prioritized, then energy):")
                        for i, seq in enumerate(best_accepted):
                            sequence_text = seq.get('sequence', 'N/A')
                            memory_section.append(f"  {i+1}. Sequence: {sequence_text}")
                            memory_section.append(f"     Score: {seq['structure_score']:.2f}, Energy: {seq['energy']:.1f}")
                    
                    worst_rejected = neighboring_effects.get('sequence_similarity_analysis', {}).get('worst_rejected', [])
                    if worst_rejected and len(worst_rejected) > 0:
                        memory_section.append(f"- Worst rejected sequences to avoid:")
                        for seq in worst_rejected:
                            sequence_text = seq.get('sequence', 'N/A')
                            memory_section.append(f"  • Sequence: {sequence_text}")
                            memory_section.append(f"    Energy: {seq['energy']:.1f}, Reason: {seq.get('reason', 'unknown')}")
                    
                    local_effects = neighboring_effects.get('local_mutation_effects', {})
                    if local_effects and len(local_effects) > 0:
                        memory_section.append(f"- Local mutation patterns near position {agent_input['position'] + 1}:")
                        if local_effects.get('successful_nearby_mutations'):
                            successful = dict(list(local_effects['successful_nearby_mutations'].items())[:3])
                            memory_section.append(f"  • Successful nearby: {successful}")
                        if local_effects.get('failed_nearby_mutations'):
                            failed = dict(list(local_effects['failed_nearby_mutations'].items())[:3])
                            memory_section.append(f"  • Failed nearby: {failed}")
                
                if iteration_details and iteration_details.get('full_iteration_history'):
                    
                    if iteration_details.get('acceptance_patterns'):
                        patterns = iteration_details['acceptance_patterns']
                        memory_section.append(f"\nPersonal Mutations Analysis at Position {agent_input['position'] + 1}:")
                        memory_section.append(f"- Acceptance trend (first half vs second half): {patterns.get('acceptance_trend', 'unknown')}")
                        if patterns.get('rejection_reasons'):
                            reasons = patterns['rejection_reasons']
                            memory_section.append(f"- Common rejection reasons: {reasons}")
                    
                    if personal_insights.get('preferred_mutations'):
                        preferred = personal_insights['preferred_mutations']
                        memory_section.append(f"- Your successful patterns: {[f'{p[0]} ({p[1]:.1%})' for p in preferred]}")

                    if personal_insights.get('avoided_mutations'):
                        avoided = personal_insights['avoided_mutations']
                        memory_section.append(f"- Your rejected patterns: {[f'{p[0]} ({p[1]:.1%})' for p in avoided]}")
            
            if personal_insights:
                if personal_insights.get('recent_trend') is not None:
                    memory_section.append(f"- Recent trend in the last 10 iterations: {personal_insights.get('recent_trend', 'unknown')}")
                
                if position_history and position_history.get('successful_residues'):
                    successful_res = position_history['successful_residues']
                    memory_section.append(f"- Residues that were successfully accepted at this position and the number of times they were accepted: {dict(list(successful_res.items()))}")
                
                if position_history and position_history.get('rejected_residues'):
                    rejected_res = position_history['rejected_residues']
                    memory_section.append(f"- Residues that were rejected at this position and the number of times they were rejected: {dict(list(rejected_res.items()))}")
                
                if position_history and position_history.get('current_state_performance'):
                    perf = position_history['current_state_performance']
                    memory_section.append(f"- Current residue {agent_input['state']} performance: {perf['times_accepted']}/{perf['times_tried']} accepted")
                    if perf.get('avg_energy_change'):
                        memory_section.append(f"- Average total energy with {agent_input['state']}: {perf['avg_energy_change']:.2f}")
            
            energy_analysis = memory_context.get('energy_impact_analysis', {})
            if energy_analysis and global_insights.get('total_iterations', 0) > 0:
                memory_section.append(f"\nPersonal and Neighborhood Energy Analysis from Memory:")
                
                if energy_analysis.get('energy_improving_changes'):
                    improving = energy_analysis['energy_improving_changes'][:3]
                    memory_section.append(f"- Energy improving iterations:")
                    for change in improving:
                        memory_section.append(f"  • Iter {change['iteration']}: energy {change['energy']:.1f}, {change['mutation_count']} mutations, {'accepted' if change['accepted'] else 'rejected'}")
                
                if energy_analysis.get('energy_worsening_changes'):
                    worsening = energy_analysis['energy_worsening_changes'][:3]
                    memory_section.append(f"- Energy worsening iterations to avoid:")
                    for change in worsening:
                        memory_section.append(f"  • Iter {change['iteration']}: energy {change['energy']:.1f}, {change['mutation_count']} mutations, {'accepted' if change['accepted'] else 'rejected'}")

            mutation_recommendations = memory_context.get('mutation_recommendations', {})
            if mutation_recommendations and global_insights.get('total_iterations', 0) > 0:
                memory_section.append(f"\nMutation Recommendations for {agent_input['state']} at Position {agent_input['position'] + 1} from Memory:")
                
                if mutation_recommendations.get('highly_recommended'):
                    highly_rec = mutation_recommendations['highly_recommended'][:3]
                    memory_section.append(f"- Highly recommended (and past success rate): {[f'{aa} ({pattern_data})' for aa, score, pattern_data in highly_rec]}")
                
                if mutation_recommendations.get('recommended'):
                    rec = mutation_recommendations['recommended'][:3]
                    memory_section.append(f"- Recommended (and past success rate): {[f'{aa} ({pattern_data})' for aa, score, pattern_data in rec]}")
                
                if mutation_recommendations.get('avoid'):
                    avoid = mutation_recommendations['avoid'][:3]
                    memory_section.append(f"- Avoid (and past success rate): {[f'{aa} ({pattern_data})' for aa, score, pattern_data in avoid]}")

            learning_summary = memory_context.get('learning_summary', '')
            if learning_summary and global_insights.get('total_iterations', 0) > 0:
                memory_section.append(f"\nOne-line Summary from Memory:")
                memory_section.append(f"{learning_summary}")
            
            relevant_experience = memory_context.get('relevant_experience', [])
            if relevant_experience:
                memory_section.append(f"\nYour Relevant Past Experiences and Energy Changes at Position {agent_input['position'] + 1}:")
                for exp in relevant_experience[:3]:
                    outcome = "accepted" if exp.get('was_accepted') else "rejected"
                    energy_info = f" (ΔE: {exp.get('energy_change', 'N/A')})" if exp.get('energy_change') else ""
                    memory_section.append(f"- Iter {exp.get('iteration', '?')}: {exp.get('old_state', '?')}->{exp.get('new_state', '?')} → {outcome}{energy_info}")
            
            prompt_parts.extend(memory_section)

            if global_insights.get('total_iterations', 0) > 0:
                prompt_parts.append("\nPART 4: Design Goal and Energy Analysis")

            iteration_details = memory_context.get('recent_iteration_details', {})
            if iteration_details and iteration_details.get('energy_progression'):
                energy_progression = iteration_details['energy_progression']
                structure_analysis = energy_progression.get('structure_energy_analysis', {})
                
                if structure_analysis.get('current_vs_best_same_score'):
                    same_score_analysis = structure_analysis['current_vs_best_same_score']
                    memory_section.append(f"Design Goal versus Energy Optimization:")
                    memory_section.append(f"- Current structure score: {same_score_analysis['current_score']:.1f}/100")
                    memory_section.append(f"- Current energy: {same_score_analysis['current_energy']:.1f}")
                    memory_section.append(f"- Best energy achieved with this score: {same_score_analysis['best_energy_same_score']:.1f}")
                    
                    if same_score_analysis['has_improvement_potential']:
                        energy_gap = same_score_analysis['energy_gap']
                        memory_section.append(f"- Energy improvement potential: {energy_gap:.1f} units below best for this score and focus on energy-lowering mutations that maintain structure score")
                    else:
                        memory_section.append(f"- This is the best energy at this structure score {same_score_analysis['current_score']:.1f} so far and consider score-improving mutations")
                
                if structure_analysis.get('pareto_efficient_points'):
                    pareto_points = structure_analysis['pareto_efficient_points'][:3]
                    memory_section.append(f"- Pareto optimal structure-energy points:")
                    for point in pareto_points:
                        memory_section.append(f"  • Score {point['score']:.1f}, Energy {point['energy']:.1f} (iter {point['iteration']})")
        
        if structure_feedback:
            feedback_text = "\nDesign Goal Evaluation:"
            
            if 'score' in structure_feedback:
                feedback_text += f"\n- Current design goal score: {structure_feedback['score']:.1f}/100, "
            
            if 'assessment' in structure_feedback:
                feedback_text += f"{structure_feedback['assessment']}"
            
            if 'details' in structure_feedback:
                details = structure_feedback['details']
                feedback_text += f"\n- Key aspects:"
                for key, value in list(details.items()):
                    feedback_text += f"\n  • {key}: {value}"
            
            if 'recommendations' in structure_feedback:
                recommendations = structure_feedback['recommendations']
                if recommendations:
                    feedback_text += f"\nStructure and Design Goal Score Recommendations for {agent_input['state']} at Position {agent_input['position'] + 1}:"
                    for rec in recommendations:
                        feedback_text += f"\n- {rec}"
            
            prompt_parts.append(feedback_text)
            
        prompt = "\n".join(prompt_parts)
        
        print(f"\n" + "="*80)
        current_iter = memory_context.get('global_insights', {}).get('current_iteration')
        if current_iter is None:
            current_iter = memory_context.get('global_insights', {}).get('total_iterations', '?')
        print(f"\nAGENT INPUT LOG - Position {agent_input['position'] + 1} - Iteration {current_iter}")
        print(f"\nCurrent residue: {agent_input['state']}")
        print(f"\nSequence: {agent_input['sequence']}")
        
        print(f"\nPROMPT DETAILS:")
        print(f"- Total prompt length: {len(prompt)} characters")
        print(f"- Contains memory context: {'Yes' if memory_context else 'No'}")
        print(f"- Contains structure feedback: {'Yes' if structure_feedback else 'No'}")
        print(f"- Contains rejected sequences: {'Yes' if rejected_sequences else 'No'}")
        
        print(f"\n" + "="*80)
        print(f"\nFULL AGENT PROMPT:")
        print(prompt)
        print(f"\nEND OF AGENT INPUT - Position {agent_input['position'] + 1}")
        print(f"="*80)
        
        result = call_llm(
            role="protein design agent",
            query=prompt,
            response_format=DesignProposal,
            system_message="You are a protein design agent with memory and learning capabilities. You understand protein folding principles and follow decision rules to achieve design goals through conservative mutations when possible.",
            history=None
        )
        
        print(f"\nAgent @ position {agent_input['position'] + 1}")
        print("Response:\n", result)
        
        if isinstance(result, dict):
            return result["proposed_value"]
        
        import json
        import re
        
        try:
            parsed_result = json.loads(result)
            return parsed_result["proposed_value"]
        except (json.JSONDecodeError, KeyError):
            pass
        
        json_match = re.search(r'\{.*?"proposed_value".*?\}', result, re.DOTALL)
        if json_match:
            try:
                parsed_result = json.loads(json_match.group())
                return parsed_result["proposed_value"]
            except:
                pass
        
        aa_match = re.search(r'\b([ACDEFGHIKLMNPQRSTVWY])\b', result)
        if aa_match:
            return aa_match.group(1)
        
        print(f"WARNING: Could not parse response, keeping current residue")
        return agent_input['state']
    
    finally:
        _IN_AGENT_PROCESSING = False 