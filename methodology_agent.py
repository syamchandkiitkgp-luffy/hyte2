import sys
import os
import re
import json
from state import GraphState
from observability import trace_node

# Add Data_Dictionary to path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
if os.path.join(os.getcwd(), 'Data_Dictionary') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))

from gemini_client import call_gemini
from rag_retriever import RAGRetriever
from pseudocode_agent import PseudocodeAgent

"""
### Concept: Multi-Persona Review (Self-Correction)
One of the best ways to improve LLM output is to simulate a team of experts collaborating. 
This is called the **Multi-Persona** pattern.

In the Methodology Agent, we use three distinct roles:
1. **Consultant (Drafting)**: Focuses on creativity and strategic frameworks (McKinsey/Bain style).
2. **Lead Consultant (Critique)**: Focuses on identifying gaps, logical errors, and constraint violations.
3. **Principal Consultant (Finalization)**: Weighs the draft against the critique to produce a polished, high-quality final plan.

#### Why use multiple personas?
- **Reduced Hallucinations**: Having a separate "Critique" phase forces the AI to check its own work.
- **Higher Quality**: It separates "idea generation" from "quality control," mirroring how high-stakes professional work is done.
"""

class MethodologyAgent:
    """
    Agent responsible for designing the analysis strategy.
    Uses a multi-stage process: Data-Agnostic Strategy -> Data Retrieval -> Data-Aware Methodology.
    """
    
    @trace_node("Methodology")
    def run(self, state: dict):
        hypothesis = state.get("hypothesis")
        current_step = state.get("current_step")
        
        # Stage 1: Generate initial data-agnostic strategy
        if current_step in ["trigger_initial_strategy", "refine_strategy"]:
            if current_step == "refine_strategy" and state.get("initial_strategy") and state.get("latest_feedback"):
                strategy_output = self.refine(hypothesis, None, state["initial_strategy"], state["latest_feedback"])
                # strategy_output is now a formatted string from self.refine if we follow the format_draft(refined_dict)
                # But we need the dict for the next stage.
                # Actually, self.refine returns self._format_draft(refined_dict)
                # We need to re-parse it or change refine to return both.
                # Let's update refine to return the dict as well if possible, or just parse it here.
                strategy = strategy_output
                strategy_dict = self.parse_sections(strategy)
            else:
                strategy_dict = self._get_consultant_draft(hypothesis, data_context=None)
                critique = self._get_lead_critique(hypothesis, strategy_dict)
                strategy_dict = self._get_refined_draft(hypothesis, strategy_dict, critique)
                strategy = self._format_draft(strategy_dict)

            return {
                "initial_strategy": strategy,
                "initial_strategy_dict": strategy_dict,
                "current_step": "strategy_generated",
                "messages": [{
                    "role": "assistant",
                    "content": f"## 📊 Initial Analysis Strategy\n\n{strategy}\n\n---\n\n✅ **Please review the strategy above.**\n\nThis is a high-level approach before we identify specific data tables. Type 'approve' or 'yes' to proceed to data retrieval, or provide feedback to refine."
                }]
            }
        
        # 2. Stage 2: Data-Aware Methodology
        elif current_step in ["trigger_final_methodology", "refine_methodology"]:
            strategy = state.get("initial_strategy", "")
            
            # Check if it's a refinement
            if current_step == "refine_methodology" and state.get("methodology") and state.get("latest_feedback"):
                current_methodology = state["methodology"]
                metadata_context = state.get("metadata_context", "")
                feedback = state["latest_feedback"]
                # For refinement, we use the existing metadata but can refine the methodology
                result = self.refine(hypothesis, metadata_context, current_methodology, feedback)
                # If refine returns a string, wrap it. (MethodologyAgent.refine currently returns string)
                if isinstance(result, str):
                    updates = {"methodology": result}
                else:
                    updates = result
            else:
                # This now returns a dict with methodology, pseudocode, metadata_context
                strategy_dict = state.get("initial_strategy_dict")
                if not strategy_dict and state.get("initial_strategy"):
                    strategy_dict = self.parse_sections(state["initial_strategy"])
                
                result = self.generate_data_aware_methodology(hypothesis, strategy_dict)
                updates = result
            
            final_methodology = updates.get("methodology")
            meth_pseudocode = updates.get("methodology_pseudocode", state.get("methodology_pseudocode", ""))
            pseudocode_dict = updates.get("pseudocode", state.get("pseudocode", {}))
            metadata_table = updates.get("metadata_context", state.get("metadata_context", ""))
            
            # Perform feasibility analysis for per-KPI dataset generation
            kpis_dict = updates.get("kpis_dict", {})
            
            granularity_analysis = self._perform_feasibility_analysis(
                hypothesis, 
                kpis_dict, 
                metadata_table
            )
            
            final_msg = f"## 📋 Final Methodology (Data-Aware)\n\n{final_methodology}\n\n### 🧩 Master Dataset Logic (Draft)\n\n```\n{meth_pseudocode}\n```\n\n### 🔍 Selected Metadata\n\n{metadata_table}\n\n### 🔬 Feasibility Analysis\n\n{self._format_feasibility_analysis(granularity_analysis)}\n\n---\n\n✅ **Please review the integrated plan above.**\n\nType 'approve' or 'yes' to proceed to code generation, or provide feedback to refine any part."

            return {
                "methodology": final_methodology,
                "methodology_pseudocode": meth_pseudocode,
                "pseudocode": pseudocode_dict,
                "metadata_context": metadata_table,
                "granularity_analysis": granularity_analysis,
                "current_step": "methodology_generated",
                "messages": [{"role": "assistant", "content": final_msg}]
            }

    def generate_initial_strategy(self, hypothesis):
        """Stage 1: Generate data-agnostic strategy (high-level approach)"""
        draft_dict = self._get_consultant_draft(hypothesis, data_context=None)
        critique = self._get_lead_critique(hypothesis, draft_dict)
        refined_draft_dict = self._get_refined_draft(hypothesis, draft_dict, critique)
        strategy = self._format_draft(refined_draft_dict)
        return strategy, refined_draft_dict
    
    def generate_data_aware_methodology(self, hypothesis, initial_strategy_dict):
        """Stage 2: Generate methodology with actual data tables and pseudocode."""
        return self._get_data_strategy_manager_final(hypothesis, initial_strategy_dict)

    def _generate_approach(self, hypothesis, data_context=None):
        """Step 1: Generate Approach."""

        context_instruction = f"Context: {data_context}" if data_context else "Context: Data not yet retrieved. Focus on the STRATEGY."
        
        prompt = f"""
        You are an expert Data Science and Strategy Consultant.
        Draft the **Approach** section for the hypothesis: "{hypothesis}"
        
        {context_instruction}
        
        Task:
        - Identify at max 2-3 real world scenarios that impact the core intent of the hypothesis.
        - Provide a comprehensive strategy to study those scenarios & hypothesis and its potential impact.
        - The objective of the strategy should be to develop a step by step plan which is measurable and testable to provide a clear and concise actionalble insights to the hypothesis.
        - If required, use the frameworks and best practices used by the McKinsey, Bain, BCG, GE or top teir consulting firms etc. for data driven decision making.
        - List down the steps of how to recreate, study, analyze the real world scenarios with the possible outcomes of accepting or rejecting the hypothesis.

        Constraints:
        - Max 5 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the bullet points for the Approach.
        """
        return call_gemini(prompt)

    def _generate_assumptions(self, hypothesis, approach):
        """Step 2: Generate Assumptions."""

        prompt = f"""
        You are an expert Data Science Consultant.
        Based on the Hypothesis and Approach, list the Assumptions and Clarifications to make the approach actionable more clear.
        
        Hypothesis: "{hypothesis}"
        Approach: "{approach}"
        
        Task:
        - List assumptions made while creating the methodology.
        - Ask clarifying questions if ambiguities exist.
        
        Constraints:
        - Max 4 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the bullet points.
        """
        return call_gemini(prompt)

    def _generate_kpis(self, hypothesis, approach, assumptions):
        """Step 3: Generate KPIs."""

        prompt = f"""
        You are an expert Data Science Consultant.
        Identify the KPIs required to test the hypothesis based on the approach and assumptions.
        
        Hypothesis: "{hypothesis}"
        Approach: "{approach}"
        Assumptions: "{assumptions}"
        
        Task:
        Identify a list of a maximum of 5-6 KPIs.
        - KPIs should be relevant to the hypothesis.
        - KPIs should be measurable.
        - KPIs should be actionable.
        
        Constraints:
        - Return ONLY a JSON dictionary where keys are KPI names and values are descriptions.
        - Description must be < 20 words.
        
        Example Output:
        {{
          "ARPU": "Monthly average revenue per user to track monetization.",
          "Churn Rate": "Percentage of subscribers leaving the network monthly."
        }}
        """
        return call_gemini(prompt)

    def _generate_visualizations(self, hypothesis, approach, assumptions, kpis):
        """Step 4: Generate Visualizations."""

        prompt = f"""
        You are an expert Data Science Consultant.
        Suggest Visualizations to monitor the KPIs based on the approach and assumptions.
        
        Hypothesis: "{hypothesis}"
        Approach: "{approach}"
        Assumptions: "{assumptions}"
        KPIs: "{kpis}"
        
        Task:
        - List the Visualizations.
        - Visualizations should be relevant to the KPIs & hypothesis.
        - Visualizations should be measurable.
        - Visualizations should be actionable.
        
        Constraints:
        - Max 5-6 Visualizations.
        - **Format**: Return a JSON-style Array List where each item is a Key-Value pair:
          - Key: Chart Type with Dimensions (e.g., 'Line Chart (Date vs Churn)')
          - Value: Description and Intent (MUST be < 15 words).
        
        Example Output:
        - Line Chart (Date vs Revenue): Trends in revenue over the last 12 months.
        - Bar Chart (Region vs Churn): Comparing churn rates across different geographic regions.
        
        Output format:
        Return ONLY the list.
        """
        return call_gemini(prompt)

    def _get_consultant_draft(self, hypothesis, data_context=None):
        """Phase 1: Data Science and Strategy Consultant prepares a draft using sequential steps."""

        
        approach = self._generate_approach(hypothesis, data_context)
        assumptions = self._generate_assumptions(hypothesis, approach)
        kpis = self._generate_kpis(hypothesis, approach, assumptions)
        visualizations = self._generate_visualizations(hypothesis, approach, assumptions, kpis)
        
        return {
            "approach": approach,
            "assumptions": assumptions,
            "kpis": kpis,
            "visualizations": visualizations
        }

    def _format_draft(self, draft_dict):
        """Helper to format the draft dictionary into markdown."""
        return f"""
                1. **Approach**:
                {draft_dict['approach']}

                2. **KPIs**:
                {draft_dict['kpis']}

                3. **Visualizations**:
                {draft_dict['visualizations']}

                4. **Assumptions/Clarifications Needed**:
                {draft_dict['assumptions']}
                """

    def _get_lead_critique(self, hypothesis, draft):
        """Phase 2: Lead Consultant provides feedback on the draft."""

        
        # Ensure draft is string for prompt
        draft_text = draft if isinstance(draft, str) else self._format_draft(draft)
        
        prompt = f"""
        You are a Lead Data Science Consultant. Critique this draft for hypothesis: "{hypothesis}"
        Draft: {draft_text}
        
        Task:
        Provide critical feedback on the draft methodology for each of the sections seperately. Focus on the strategic value, logic, and adherence to constraints.
        Approach:
        1. Strategy Alignment: Are the scenarios identified relevant to the hypothesis? Does the methodology effectively test the hypothesis?
        2. Logic & Soundness: Is the analytical approach sound?
        3. Clarity: Does the approach has clear and understandable steps?
        4. Completeness: Does the approach cover all the necessary steps to test the hypothesis?
        
        Assumptions/Clarifications:
        1. Assumptions/Clarifications: Are the assumptions and clarifying questions relevant and helpful?
        
        KPIs:
        1. KPIs: Are there max 5-6 relevant KPIs?
        2. KPIs: Are the KPIs relevant to the hypothesis?
        3. KPIs: Are the KPIs measurable?
        4. KPIs: Are the KPIs actionable?
        
        Visualizations:
        1. Visualizations: Are there max 5-6 relevant visualizations?
        2. Visualizations: Are the visualizations relevant to the KPIs & hypothesis?
        3. Visualizations: Are the visualizations measurable?
        4. Visualizations: Are the visualizations actionable?
        
        Provide clear, actionable feedback points. Do not rewrite the methodology, just critique it.
        """
        return call_gemini(prompt)

    def _get_refined_draft(self, hypothesis, draft_dict, critique):
        """Phase 3: Principal Consultant creates the refined plan sequentially."""

        
        refined_approach = self._refine_approach(hypothesis, draft_dict['approach'], critique)
        refined_assumptions = self._refine_assumptions(hypothesis, refined_approach, draft_dict['assumptions'], critique)
        refined_kpis = self._refine_kpis(hypothesis, refined_approach, refined_assumptions, draft_dict['kpis'], critique)
        refined_visualizations = self._refine_visualizations(hypothesis, refined_approach, refined_assumptions, refined_kpis, draft_dict['visualizations'], critique)
        
        return {
            "approach": refined_approach,
            "assumptions": refined_assumptions,
            "kpis": refined_kpis,
            "visualizations": refined_visualizations
        }

    def _refine_approach(self, hypothesis, original_approach, critique):
        """Step 3.1: Refine Approach."""

        prompt = f"""
        You are a Principal Data Science Consultant. Refine the Approach based on the Lead Consultant's critique.
        
        Hypothesis: "{hypothesis}"
        Original Approach: "{original_approach}"
        Critique: "{critique}"
        
        Task:
        Refine the Approach section to address the critique while staying within constraints.
        - Identify at max 2-3 real world scenarios that impact the core intent of the hypothesis.
        - Provide a comprehensive strategy to study those scenarios & hypothesis and its potential impact.
        - The objective of the strategy should be to develop a step by step plan which is measurable and testable to provide a clear and concise actionalble insights to the hypothesis.
        - If required, use the frameworks and best practices used by the McKinsey, Bain, BCG, GE or top teir consulting firms etc. for data driven decision making.
        - List down the steps of how to recreate, study, analyze the real world scenarios with the possible outcomes of accepting or rejecting the hypothesis.

        Constraints:
        - Max 6 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined Approach bullet points.
        """
        return call_gemini(prompt)

    def _refine_assumptions(self, hypothesis, refined_approach, original_assumptions, critique):
        """Step 3.2: Refine Assumptions."""

        prompt = f"""
        You are a Principal Data Science Consultant. Refine the Assumptions/Clarifications based on the refined Approach and Lead Consultant's critique to make the approach actionable and more clear.
        
        Hypothesis: "{hypothesis}"
        Refined Approach: "{refined_approach}"
        Original Assumptions: "{original_assumptions}"
        Critique: "{critique}"
        
        Task:
        - Refine the Assumptions/Clarifications section to address the critique and align with the refined approach.
        - List assumptions made while creating the methodology.
        - Ask clarifying questions if ambiguities exist.
        
        Constraints:
        - Max 4 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined bullet points.
        """
        return call_gemini(prompt)

    def _refine_kpis(self, hypothesis, refined_approach, refined_assumptions, original_kpis, critique):
        """Step 3.3: Refine KPIs with validation."""

        
        for attempt in range(3):
            prompt = f"""
            You are a Principal Data Science Consultant. Refine the KPIs based on the refined Approach, refined Assumptions, and Lead Consultant's critique.
            
            Hypothesis: "{hypothesis}"
            Refined Approach: "{refined_approach}"
            Refined Assumptions: "{refined_assumptions}"
            Original KPIs: "{original_kpis}"
            Critique: "{critique}"
            
            Task:
            Identify a list of a maximum of 5-6 refined KPIs. Address the critique and align with the refined strategy.
            - KPIs should be relevant to the hypothesis.
            - KPIs should be measurable.
            - KPIs should be actionable.
            
            Constraints:
            - Return ONLY a JSON dictionary where keys are KPI names and values are descriptions.
            - Description must be < 20 words.
            
            Example Output:
            {{
              "ARPU": "Monthly average revenue per user to track monetization.",
              "Churn Rate": "Percentage of subscribers leaving the network monthly."
            }}
            """
            response = call_gemini(prompt)
            
            # Validation logic
            try:
                # Basic JSON check for dictionary
                data = json.loads(re.sub(r'```json\n?|\n?```', '', response).strip())
                if isinstance(data, dict) and len(data) > 0:
                    return response
            except:
                pass
                
        return original_kpis # Fallback

    def _refine_visualizations(self, hypothesis, refined_approach, refined_assumptions, refined_kpis, original_visuals, critique):
        """Step 3.4: Refine Visualizations with validation."""

        
        for attempt in range(3):
            prompt = f"""
            You are a Principal Data Science Consultant. Refine the Visualizations based on the refined sections and Lead Consultant's critique to monitor the KPIs.
            
            Hypothesis: "{hypothesis}"
            Refined Approach: "{refined_approach}"
            Refined Assumptions: "{refined_assumptions}"
            Refined KPIs: "{refined_kpis}"
            Critique: "{critique}"
            
            Task:
            - List 5-6 refined Visualizations. Address the critique and ensure relevance to the refined KPIs.
            - Visualizations should be relevant to the KPIs & hypothesis.
            - Visualizations should be measurable.
            - Visualizations should be actionable.
            
            Constraints:
            - Max 5-6 Visualizations.
            - **Format**: Return a JSON-style Array List where each item is a Key-Value pair:
              - Key: Chart Type with Dimensions (e.g., 'Line Chart (Date vs Churn)')
              - Value: Description and Intent (MUST be < 15 words).
            
            Output format:
            Return ONLY the list.
            """
            response = call_gemini(prompt)
            
            # Validation logic: Check if it looks like a list with at least one item
            if response and (("-" in response) or ("[" in response)):
                return response

            
        return original_visuals # Fallback

    def _generate_data_aware_approach(self, hypothesis, strategy_dict, data_context):
        """Step 4.1: Generate Data-Aware Approach."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the Approach to reference specific tables and columns.
        
        Hypothesis: "{hypothesis}"
        Initial Strategy Approach: "{strategy_dict['approach']}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        1. Review the Strategy and the Data Context.
        2. Replace generic data references with specific Table and Column names.
        3. Ensure the approach remains strategic while becoming feasible and testable.
        
        Constraints:
        - Max 5 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined Data-Aware Approach bullet points.
        """
        return call_gemini(prompt)

    def _generate_data_aware_assumptions(self, hypothesis, strategy_dict, refined_approach, data_context):
        """Step 4.2: Generate Data-Aware Assumptions."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the Assumptions to be data-specific and actionable.
        
        Hypothesis: "{hypothesis}"
        Initial Strategy Assumptions: "{strategy_dict['assumptions']}"
        Refined Data-Aware Approach: "{refined_approach}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        1. List data-specific assumptions based on the available metadata.
        2. Ask clarifying questions regarding data quality or availability if ambiguities exist.
        
        Constraints:
        - Max 4 bullet points.
        - Max 200 words.
        
        Output format:
        Return ONLY the refined bullet points.
        """
        return call_gemini(prompt)

    def _generate_data_aware_kpis(self, hypothesis, strategy_dict, refined_approach, refined_assumptions, data_context):
        """Step 4.3: Generate Data-Aware KPIs."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the KPIs to map to specific source columns and tables.
        
        Hypothesis: "{hypothesis}"
        Initial Strategy KPIs: "{strategy_dict['kpis']}"
        Refined Data-Aware Approach: "{refined_approach}"
        Refined Data-Aware Assumptions: "{refined_assumptions}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        Identify a list of a maximum of 5-6 refined KPIs.
        - Replace generic KPIs with those that can be calculated from the provided tables.
        - Specify the source table/column for each KPI.
        
        Constraints:
        - Return ONLY a JSON dictionary where keys are KPI names and values are descriptions.
        - Description must include the source table/column.
        
        Output format:
        Return ONLY the JSON dictionary.
        """
        return call_gemini(prompt)

    def _generate_data_aware_visualizations(self, hypothesis, strategy_dict, refined_approach, refined_assumptions, refined_kpis, data_context):
        """Step 4.4: Generate Data-Aware Visualizations."""

        prompt = f"""
        You are the Data Strategy Manager. Refine the Visualizations to monitor the data-aware KPIs.
        
        Hypothesis: "{hypothesis}"
        Refined Data-Aware Approach: "{refined_approach}"
        Refined Data-Aware Assumptions: "{refined_assumptions}"
        Refined Data-Aware KPIs: "{refined_kpis}"
        Data Context (Tables & Columns):
        {data_context}
        
        Task:
        - List 5-6 refined Visualizations feasible with the available data.
        - Ensure relevance to the refined KPIs.
        
        Constraints:
        - Max 5-6 Visualizations.
        - **Format**: Return a JSON-style Array List where each item is a Key-Value pair:
          - Key: Chart Type with Dimensions (e.g., 'Line Chart (Date vs Churn)')
          - Value: Description and Intent.
        
        Output format:
        Return ONLY the list.
        """
        return call_gemini(prompt)

    def _get_data_strategy_manager_final(self, hypothesis, strategy_dict, feedback=""):
        """
        Phase 4: Data Strategy Manager - Orchestrates the full data-aware flow sequentially.
        """

        
        # 1. RAG Retrieval (Internal)

        # Use keys from the refined strategy's KPI dictionary
        refined_kpis = strategy_dict.get('kpis', {})
        if isinstance(refined_kpis, str):
            # If it's still a JSON string, parse it
            try:
                refined_kpis = json.loads(re.sub(r'```json\n?|\n?```', '', refined_kpis).strip())
            except:
                refined_kpis = {}
        
        kpi_names = list(refined_kpis.keys()) if isinstance(refined_kpis, dict) else []
        
        retriever = RAGRetriever()
        kpi_candidates = retriever.retrieve_candidates_for_kpis(kpi_names)
        
        # 2. Pseudocode & Metadata Generation (Internal)

        ps_agent = PseudocodeAgent()
        initial_strategy_text = self._format_draft(strategy_dict)
        pseudocode, context_list = ps_agent.generate_with_validation(hypothesis, initial_strategy_text, kpi_candidates, feedback=feedback)
        
        # Convert context_list to markdown table for the prompt
        def to_markdown(ctx_list):
            val = "| S.No | Table | KPIs | Columns | Reasoning |\n|---|---|---|---|---|\n"
            for row in ctx_list:
                if len(row) >= 5:
                    val += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
            return val
        data_context_table = to_markdown(context_list)
        
        # Pack data_context into strategy_dict for sub-functions
        strategy_dict['data_context'] = data_context_table

        # 3. Sequential Data-Aware Generation
        da_approach = self._generate_data_aware_approach(hypothesis, strategy_dict, data_context_table)
        da_assumptions = self._generate_data_aware_assumptions(hypothesis, strategy_dict, da_approach, data_context_table)
        da_kpis = self._generate_data_aware_kpis(hypothesis, strategy_dict, da_approach, da_assumptions, data_context_table)
        da_visualizations = self._generate_data_aware_visualizations(hypothesis, strategy_dict, da_approach, da_assumptions, da_kpis, data_context_table)
        
        final_dict = {
            "approach": da_approach,
            "assumptions": da_assumptions,
            "kpis": da_kpis,
            "visualizations": da_visualizations
        }
        final_methodology = self._format_draft(final_dict)
        
        # Parse KPIs into dict for feasibility analysis
        try:
            kpis_json = re.sub(r'```json\n?|\n?```', '', da_kpis).strip()
            kpis_dict = json.loads(kpis_json)
        except Exception as e:

            kpis_dict = {}

        return {
            "methodology": final_methodology,
            "methodology_pseudocode": pseudocode,
            "pseudocode": {},  # Initialize as empty dict for per-KPI logic
            "metadata_context": data_context_table,
            "context_list": context_list,
            "kpis_dict": kpis_dict
        }

    def generate(self, hypothesis):
        # 1. Standard Consultant Flow (Sequential)
        draft_dict = self._get_consultant_draft(hypothesis, data_context=None)
        critique = self._get_lead_critique(hypothesis, draft_dict)
        refined_strategy_dict = self._get_refined_draft(hypothesis, draft_dict, critique)
        
        # 2. Data Strategy Manager Refinement (Stage 2)
        final_plan = self._get_data_strategy_manager_final(hypothesis, refined_strategy_dict, feedback="")
        
        return final_plan

    def refine(self, hypothesis, data_context, current_methodology, feedback):
        # For refinement based on user feedback, we can use the same sequential refinement flow
        # mapping the collective feedback into the "critique"

        
        # Parse current_methodology back into a draft_dict for sequential refinement
        sections = self.parse_sections(current_methodology)
        draft_dict = {
            "approach": sections.get("meth_approach", ""),
            "assumptions": sections.get("meth_assumptions", ""),
            "kpis": sections.get("meth_kpis", ""),
            "visualizations": sections.get("meth_visualizations", "")
        }
        
        critique = self._get_lead_critique(hypothesis, current_methodology)
        combined_feedback = f"{critique}\n\nUSER FEEDBACK: {feedback}"
        
        # Instead of principal final, we can use the sequential refinement flow if we want consistency
        refined_dict = self._get_refined_draft(hypothesis, draft_dict, combined_feedback)
        
        # Parse KPIs into dict for feasibility analysis
        da_kpis = refined_dict.get('kpis', '{}')
        try:
            kpis_json = re.sub(r'```json\n?|\n?```', '', da_kpis).strip()
            kpis_dict = json.loads(kpis_json)
        except Exception as e:

            kpis_dict = {}

        return {
            "methodology": self._format_draft(refined_dict),
            "kpis_dict": kpis_dict
        }

    @staticmethod
    def parse_sections(methodology_text):
        sections = {
            "meth_approach": "",
            "meth_kpis": "",
            "meth_visualizations": "",
            "meth_assumptions": ""
        }
        # Updated patterns to match the new 4-section structure:
        # 1. **Approach**:
        # 2. **KPIs**:
        # 3. **Visualizations**:
        # 4. **Assumptions/Clarifications Needed**:
        patterns = {
            "meth_approach": r"1\.\s*\*\*Approach\*\*:(.*?)(?=2\.\s*\*\*KPIs|$)",
            "meth_kpis": r"2\.\s*\*\*KPIs\*\*:(.*?)(?=3\.\s*\*\*Visualizations|$)",
            "meth_visualizations": r"3\.\s*\*\*Visualizations\*\*:(.*?)(?=4\.\s*\*\*Assumptions|$)",
            "meth_assumptions": r"4\.\s*\*\*Assumptions/Clarifications Needed\*\*:(.*)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, methodology_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        return sections
    
    def _perform_feasibility_analysis(self, hypothesis, kpis_dict, metadata_context):
        """
        Analyze feasibility of aligning all KPI datasets to the same granularity.
        Returns a dict with granularity, mergeable groups, and joining keys.
        """

        
        kpi_names = list(kpis_dict.keys()) if kpis_dict else []
        if not kpi_names:
            return {
                "common_granularity": "unknown",
                "mergeable_groups": [],
                "joining_keys": {},
                "analysis_notes": "No KPIs identified for analysis"
            }
        
        prompt = f"""
        You are a Principal Data Architect analyzing a hypothesis testing workflow.
        
        Hypothesis: "{hypothesis}"
        
        KPIs to Calculate:
        {json.dumps(kpis_dict, indent=2)}
        
        Available Metadata (Tables and Columns):
        {metadata_context[:2000]}  # Truncate for token limit
        
        Task:
        Analyze if all KPI datasets can be aligned to the same granularity level and identify optimal merge strategy.
        
        Questions to Answer:
        1. What is the most appropriate common granularity? (e.g., customer-level, transaction-level, tower-level, daily-level)
        2. Which KPIs can share the same master dataset (mergeable)?
        3. What are the common joining keys for each mergeable group?
        4. Which KPIs require separate datasets due to incompatible granularity?
        
        Return ONLY a JSON object:
        {{
          "common_granularity": "customer-level",  // Most common granularity across KPIs
          "mergeable_groups": [
            {{"group_id": 1, "kpis": ["Churn Rate", "ARPU"], "granularity": "customer-level"}},
            {{"group_id": 2, "kpis": ["Network Uptime"], "granularity": "tower-daily-level"}}
          ],
          "joining_keys": {{
            "group_1": ["cust_id", "msisdn"],
            "group_2": ["tower_id", "date"]
          }},
          "analysis_notes": "Brief explanation of the grouping strategy"
        }}
        """
        
        try:
            response = call_gemini(prompt).strip().replace("```json", "").replace("```", "")
            analysis = json.loads(response)

            return analysis
        except Exception as e:

            # Fallback: treat all KPIs as separate
            return {
                "common_granularity": "mixed",
                "mergeable_groups": [{
                    "group_id": i+1,
                    "kpis": [kpi],
                    "granularity": "unknown"
                } for i, kpi in enumerate(kpi_names)],
                "joining_keys": {},
                "analysis_notes": f"Fallback: treating all KPIs separately due to analysis error: {str(e)}"
            }
    
    def _format_feasibility_analysis(self, analysis):
        """Format feasibility analysis for display."""
        if not analysis:
            return "No analysis available."
        
        mergeable_groups = analysis.get("mergeable_groups", [])
        joining_keys = analysis.get("joining_keys", {})
        
        output = f"**Common Granularity:** {analysis.get('common_granularity', 'Unknown')}\\n\\n"
        output += f"**Mergeable Groups:** {len(mergeable_groups)}\\n\\n"
        
        for group in mergeable_groups:
            group_id = group.get("group_id", "?")
            kpis = ", ".join(group.get("kpis", []))
            granularity = group.get("granularity", "unknown")
            keys = ", ".join(joining_keys.get(f"group_{group_id}", []))
            output += f"- **Group {group_id}**: {kpis} ({granularity})\\n"
            if keys:
                output += f"  - Joining Keys: `{keys}`\\n"
        
        output += f"\\n**Analysis Notes:** {analysis.get('analysis_notes', 'N/A')}"
        return output
    

