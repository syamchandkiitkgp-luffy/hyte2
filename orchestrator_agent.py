import os
import json
import sys
from state import GraphState
from observability import trace_node

# Add Data_Dictionary to path
if os.path.join(os.getcwd(), 'Data_Dictionary') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))
from gemini_client import call_gemini

"""
### Concept: The Orchestrator Pattern
In a Multi-Agent System (MAS), the **Orchestrator** acts like a manager or a conductor. 
Its job isn't to do the heavy lifting (like generating code or retrieving data), but to:

1. **Understand Intent**: Interpret what the user wants based on their message and the current context.
2. **Control Flow**: Decide which specialized agent should work next.
3. **Maintain Context**: Ensure the conversation feels continuous and that everyone has the information they need.

#### Why use an Orchestrator?
- **User Experience**: The user talks to *one* AI, even though multiple specialized engines are working behind the scenes.
- **Reliability**: It can catch errors or vague requests before they reach the more expensive analysis steps.
"""

class OrchestratorAgent:
    """
    The brain of the conversational interface.
    It manages the hypothesis testing lifecycle by routing and validating user input.
    """
    
    def _validate_hypothesis(self, hypothesis, refinement_count):
        """Helper to validate hypothesis using Gemini."""

        
        validation_prompt = f"""
        Analyze this hypothesis for a Telecom Data Analysis task: "{hypothesis}"
        
        Check for:
        1. Clarity: Is it specific?
        2. Completeness: Does it imply a measurable goal?
        3. Feasibility: Is it a valid hypothesis to test?
        
        Return ONLY a JSON with:
        {{
            "is_valid": boolean,
            "missing_info": "concise description of what is missing, or empty string",
            "clarification_question": "question to ask user, or empty string"
        }}
        """
        try:
            validation_res = json.loads(call_gemini(validation_prompt).strip().replace("```json", "").replace("```", ""))
            
            if validation_res["is_valid"] or refinement_count >= 3:
                warning = "\n_(Proceeding despite potential ambiguity due to max retries)_" if refinement_count >= 3 else ""
                return {
                    "valid": True,
                    "message": f"Hypothesis confirmed.{warning} I'll now design an initial analysis strategy...",
                    "next_step": "trigger_initial_strategy"  # NEW: Start with initial strategy
                }
            else:
                return {
                    "valid": False,
                    "message": f"I need a bit more clarity before we start.\n\n{validation_res['clarification_question']}",
                    "next_step": "hypothesis_refinement" # Stay in this step
                }
        except Exception as e:

            # Fallback to proceed
            return {"valid": True, "message": "Hypothesis recorded. Proceeding...", "next_step": "trigger_initial_strategy"}

    @trace_node("Orchestrator")
    def run(self, state: GraphState):
        """Processes the state and generates the orchestrator's response or decision."""
        messages = state.get("messages", [])
        hypothesis = state.get("hypothesis", "")
        current_step = state.get("current_step", "start")
        refinement_count = state.get("refinement_count", 0)
        
        # 1. Handle empty hypothesis
        if not hypothesis and current_step == "start":
            welcome_msg = "Welcome to **HyTE (Hypothesis Testing Engine)**! 📊\n\nI can help you analyze telecom data to validate or reject any hypothesis. Please share your hypothesis to get started.\n\n_Example: Check if customer churn increased in Q1 2024 due to network outages._"
            return {"messages": [{"role": "assistant", "content": welcome_msg}]}

        # 2. Identify turn type
        is_user_turn = messages[-1]["role"] == "user" if messages else False
        last_message = messages[-1]["content"] if messages else ""

        # State Mapping - what happens when user approves/refines at each step
        APPROVAL_MAP = {
            "strategy_generated": ("trigger_final_methodology", "refine_strategy"),
            "methodology_generated": ("initialize_kpi_workflow", "refine_methodology"),
            "pseudocode_generated": ("next_kpi_pseudocode", "refine_pseudocode"),  # Route to loop handler
            "pseudocode_review": ("trigger_codegen_workflow", "refine_pseudocode"),    # Batch approval for all pseudocode
            "code_generated": ("next_kpi_codegen", "refine_codegen"),             # Route to loop handler
            "code_review": ("trigger_execution", "refine_codegen"),                  # Batch approval for all code
            "executed": ("trigger_merge", None),
            "merge_completed": ("trigger_evaluation", None)
        }

        # 3. SPECIAL LOGIC for 'start' -> 'hypothesis_refinement' transition
        # If user just sent the hypothesis (start -> refinement), we must VALIDATE immediately.
        if current_step == "start" and is_user_turn:
            # Update hypothesis with the user's input
            new_hypothesis = last_message
            validation = self._validate_hypothesis(new_hypothesis, 0)
            
            return {
                "hypothesis": new_hypothesis,
                "current_step": validation["next_step"], 
                "refinement_count": 0 if validation["valid"] else 1,
                "messages": [{"role": "assistant", "content": validation["message"]}]
            }

        # 4. SPECIAL LOGIC for 'hypothesis_refinement' loop
        if current_step == "hypothesis_refinement" and is_user_turn:
            # User replied to clarification. Update hypothesis context?
            # Append refinement to hypothesis context for better validation
            # Or just treat the latest message as the clarification.
            # Let's append if it's not a complete replacement. 
            # Simplification: Append to a "hypothesis" string or just use Gemini to merge?
            # Let's just append for context in prompt, but maybe not overwrite state["hypothesis"] cleanly?
            # Better: Update state["hypothesis"] to include the clarification.
            
            updated_hypothesis = f"{hypothesis}\n(Clarification: {last_message})"
            validation = self._validate_hypothesis(updated_hypothesis, refinement_count)
            
            return {
                "hypothesis": updated_hypothesis,
                "current_step": validation["next_step"],
                "refinement_count": refinement_count + 1 if not validation["valid"] else refinement_count,
                "messages": [{"role": "assistant", "content": validation["message"]}]
            }

        # 5. Normal Action Phase (User approval/refinement for other steps)
        if is_user_turn:
            intent_prompt = f"""
            User Response: "{last_message}"
            Current Step: {current_step}
            
            Determine the user's intent:
            - "APPROVAL": They want to move to the next analysis step.
            - "REFINE": They want to modify the current output.
            - "QUERY": They are asking a question or just talking.
            
            Response: APPROVAL, REFINE, or QUERY.
            """
            intent = call_gemini(intent_prompt).strip().upper()


            if intent == "APPROVAL":
                if current_step in APPROVAL_MAP:
                    next_step = APPROVAL_MAP[current_step][0]
                    
                    # Special handling for KPI workflow steps
                    if next_step == "initialize_kpi_workflow":
                        return self._initialize_kpi_workflow(state)
                    elif next_step == "next_kpi_pseudocode":
                        return self._handle_kpi_loop(state, "pseudocode")
                    elif next_step == "next_kpi_codegen":
                        return self._handle_kpi_loop(state, "codegen")
                    elif next_step == "trigger_codegen_workflow":
                        return self._trigger_codegen_workflow(state)
                    elif next_step == "next_kpi_or_execute":
                        return self._handle_kpi_loop(state, "codegen")
                    elif next_step == "trigger_codegen_for_kpi":
                        return {"current_step": "trigger_codegen"}
                    elif next_step == "trigger_execution":
                        return {"current_step": "trigger_execution"}
                    else:
                        return {"current_step": next_step}
            elif intent == "REFINE":
                if current_step in APPROVAL_MAP and APPROVAL_MAP[current_step][1]:
                    refine_step = APPROVAL_MAP[current_step][1]
                    # Update State with Feedack
                    return {
                        "current_step": refine_step, 
                        "latest_feedback": last_message,
                        "messages": [{"role": "assistant", "content": f"Understood. I've noted your feedback: '{last_message}'.\n\nRedirecting to **{refine_step.replace('_', ' ').title()}** to update the work..."}]
                    }
        else:
            # 5.5 Auto-Advance Phase (Agent Turn)
            # If a step was just completed by an agent, check if we should loop automatically
            if current_step == "pseudocode_generated":
                return self._handle_kpi_loop(state, "pseudocode")
            if current_step == "code_generated":
                return self._handle_kpi_loop(state, "codegen")
            
            # If we reach here during orchestrator turns for review states, skip reporting
            if current_step in ["pseudocode_review", "code_review"]:
                return {}  # No additional messaging needed

        # 6. Reporting Phase (Agent Turn or Fallback)
        metadata = state.get('metadata_context', '')
        methodology = state.get('methodology', '')
        pseudocode = state.get('pseudocode', '')
        python_code = state.get('python_code', '')
        results = state.get('execution_results', '')

        prompt = f"""
        You are the HyTE Orchestrator. 
        Show the user the output of the current step and ask for approval/feedback.
        
        Current Step: {current_step}
        Last Message: {last_message}
        
        Current Context:
        - Hypothesis: {hypothesis}
        - Strategy: {'Planned' if methodology else 'Pending'}
        - Metadata: {'Identified' if metadata else 'Pending'}
        - Logic: {'Defined' if pseudocode else 'Pending'}
        - Code: {'Written' if python_code else 'Pending'}
        - Results: {'Available' if results else 'Pending'}
        
        Instructions:
        - Explain what was just accomplished.
        - Be concise.
        - ALWAYS end by asking: "Should I proceed or would you like to refine this?"
        """
        
        response = call_gemini(prompt)
        return {"messages": [{"role": "assistant", "content": response}]}
    
    def _initialize_kpi_workflow(self, state):
        """Initialize per-KPI workflow by extracting KPIs from feasibility analysis."""
        granularity_analysis = state.get("granularity_analysis", {})
        
        # Extract all KPIs from mergeable groups
        all_kpis = []
        for group in granularity_analysis.get("mergeable_groups", []):
            all_kpis.extend(group.get("kpis", []))
        
        if not all_kpis:

            return {
                "current_step": "trigger_codegen",  # Fallback to old workflow
                "messages": [{
                    "role": "assistant",
                    "content": "⚠️ No KPIs identified in feasibility analysis. Proceeding with standard workflow."
                }]
            }
        

        
        return {
            "kpi_list": all_kpis,
            "current_kpi": all_kpis[0],
            "current_kpi_index": 0,
            "current_step": "trigger_pseudocode",  # Generate pseudocode for first KPI
            "messages": [{
                "role": "assistant",
                "content": f"## 🎯 Starting Per-KPI Workflow\n\n**Total KPIs**: {len(all_kpis)}\n**KPIs**: {', '.join(all_kpis)}\n\n---\n\n**Now processing**: {all_kpis[0]}\n\nGenerating pseudocode..."
            }]
        }

    def _trigger_codegen_workflow(self, state):
        """Initialize the Codegen phase for the first KPI."""
        kpi_list = state.get("kpi_list", [])
        if not kpi_list:
             return {"current_step": "trigger_codegen"}
             
        return {
            "current_kpi": kpi_list[0],
            "current_kpi_index": 0,
            "current_step": "trigger_codegen",
            "messages": [{
                "role": "assistant",
                "content": f"## 🛠️ Starting Code Generation\n\n**Processing**: {kpi_list[0]} (1/{len(kpi_list)})\n\nGenerating Python script..."
            }]
        }

    def _handle_kpi_loop(self, state, phase):
        """Automatically advance to next KPI or stop for approval."""
        kpi_list = state.get("kpi_list", [])
        current_kpi_index = state.get("current_kpi_index", 0)
        
        if not kpi_list:
            return {"current_step": f"{phase}_review"}

        # If we just finished a KPI, check if there are more
        if current_kpi_index < len(kpi_list) - 1:
            next_index = current_kpi_index + 1
            next_kpi = kpi_list[next_index]
            
            trigger_step = "trigger_pseudocode" if phase == "pseudocode" else "trigger_codegen"
            
            return {
                "current_kpi": next_kpi,
                "current_kpi_index": next_index,
                "current_step": trigger_step,
                "messages": [{
                    "role": "assistant", 
                    "content": f"✅ {phase.title()} for **{kpi_list[current_kpi_index]}** complete. Moving to **{next_kpi}** ({next_index+1}/{len(kpi_list)})..."
                }]
            }
        else:
            # All KPIs done for this phase, stop for approval
            final_step = "pseudocode_review" if phase == "pseudocode" else "code_review"
            
            # Create a summary message with all artifacts
            summary_parts = [f"## 🏁 {phase.title()} Generation Complete\n\nAll {len(kpi_list)} KPIs have been processed.\n"]
            
            if phase == "pseudocode":
                artifacts = state.get("pseudocode", {})
                for kpi in kpi_list:
                    summary_parts.append(f"### 🧩 {kpi}\n```\n{artifacts.get(kpi, 'N/A')}\n```\n")
            else: # codegen
                artifacts = state.get("python_code", {})
                for kpi in kpi_list:
                    summary_parts.append(f"### 🐍 {kpi}\n```python\n{artifacts.get(kpi, 'N/A')}\n```\n")
            
            summary_parts.append("\n---\n\n**Please review the results above.** Type 'approve' to proceed to the next stage.")
            
            full_summary = "\n".join(summary_parts)
            
            return {
                "current_step": final_step,
                "messages": [{
                    "role": "assistant",
                    "content": full_summary
                }]
            }
    
    def _next_kpi_or_execute(self, state):
        """Advance to next KPI or trigger execution if all KPIs processed (kept for compatibility)."""
        return self._handle_kpi_loop(state, "codegen")
