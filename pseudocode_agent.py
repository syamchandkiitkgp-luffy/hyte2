import sys
import os
from state import GraphState
from observability import trace_node, trace_tool

# Add Data_Dictionary to path
if os.path.join(os.getcwd(), 'Data_Dictionary') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))
from gemini_client import call_gemini

"""
### Concept: Intermediate Logic Representation (Pseudocode)
When moving from a high-level plan (Methodology) to actual code (Python), directly generating a large script code can lead to errors. 

The **Pseudocode Agent** acts as a bridge. It creates an **Intermediate Representation**—a logical "blueprint"—that describes the data transformations in plain English before a single line of code is written.

#### Why use Pseudocode?
1. **Validation**: We can verify the logic (e.g., checking if the correct tables and columns are used) at a high level.
2. **Reviewability**: Non-technical users can review "Step 1: Load customer data" much easier than a Pandas SQL join.
3. **Consistency**: It gives the CodeGen agent a strict list of instructions, significantly reducing syntax and logic errors in the final script.
"""

class PseudocodeAgent:
    """
    Agent responsible for translating the methodology into technical logic steps.
    Includes a validation loop to ensure all referenced tables and columns exist.
    """
    
    @trace_node("Pseudocode")
    def run(self, state: GraphState):
        """Generates per-KPI pseudo-code artifact for LangGraph."""
        methodology = state["methodology"]
        data_context = state["metadata_context"]
        current_kpi = state.get("current_kpi", "")
        granularity_analysis = state.get("granularity_analysis", {})
        
        existing_pseudocode_dict = state.get("pseudocode", {})
        if not isinstance(existing_pseudocode_dict, dict):
            existing_pseudocode_dict = {}
        
        if not current_kpi:
            # Fallback: if no current_kpi, return empty
            return {
                "pseudocode": {},
                "current_step": "pseudocode_generated",
                "messages": [{"role": "assistant", "content": "Error: No KPI specified for pseudocode generation."}]
            }
        
        # Check if refinement
        if existing_pseudocode_dict.get(current_kpi) and state.get("current_step") == "refine_pseudocode":
            feedback = state.get("latest_feedback") or (state["messages"][-1]["content"] if state["messages"] else "Improve the logic.")
            pseudocode = self.refine(methodology, existing_pseudocode_dict[current_kpi], feedback)
        else:
            # Generate KPI-specific pseudocode
            pseudocode = self.generate_for_kpi(
                kpi_name=current_kpi,
                methodology=methodology,
                data_context=data_context,
                granularity_analysis=granularity_analysis
            )
        
        # Update dict
        updated_pseudocode_dict = existing_pseudocode_dict.copy()
        updated_pseudocode_dict[current_kpi] = pseudocode
        
        return {
            "pseudocode": updated_pseudocode_dict,
            "current_step": "pseudocode_generated",
            "messages": [{
                "role": "assistant",
                "content": f"✅ Pseudocode generated for **{current_kpi}**"
            }]
        }

    def generate(self, methodology, data_context, feedback="", mode='master'):
        """
        Generate pseudo code based on methodology using SDE, Lead, and Principal personas.
        """
        if mode != 'master':
            pass

        # Prepare feedback string to avoid backslash in f-string expression (Python < 3.12 compatibility)
        feedback_str = f"Validation Feedback from previous attempt:\n{feedback}\n" if feedback else ""

        # Step 1: Software Development Engineer (Drafting)

        sde_prompt = f"""
You are an expert Software Development Engineer and Python Specialist.
Your task is to create CONCISE, natural language steps to create the MASTER DATASET (Semantic Layer).
The steps should be detailed end to end process of creating the master dataset. 

Constraint: 
Use ONLY the tables and columns identified in the MASTER METADATA TABLE below.

Context:
MASTER METADATA TABLE (Grounding):
{data_context}

{feedback_str}

Methodology & Technical Approach:
{methodology}

CRITICAL REQUIREMENTS:
1. **Strict Grounding**: You MUST use the tables and columns specified in the MASTER METADATA TABLE. Do not invent others.
2. **Maximum 10 Steps**: You MUST limit the pseudo code to 10 steps or fewer.
3. **Consolidate Operations**: Combine related operations (e.g., "Load data and standardize columns" instead of two separate steps).
4. **High-Level Only**: Focus on major transformations, and always include table names and column names in each and every step.
5. **Natural Language**: No code syntax. Use plain English descriptions.
6. Each step should be in a new line and visually separated from other steps.
7. Explicitly mention table names and respective column names in the steps.
8. **Concise**: Each step max 20-25 words.
9. **Final Step**: Must save to 'master_dataset.csv'.

Guidelines for Consolidation:
- Combine: Load + Clean -> "Load and clean data"
- Combine: Filter + Merge -> "Filter events and merge with customer data"
- Combine: Multiple aggregations -> "Aggregate metrics"
- Combine: Feature engineering -> "Calculate KPIs and derived features"

Output Format:
Step 1: [Load col1, col2, ... from table_name 1 ...]
Step 2: [Consolidated action description]
...
Step N (≤10): Save the final processed dataframe to 'master_dataset.csv'.

Remember: MAXIMUM 10 STEPS TOTAL.
"""
        draft_pseudo_code = call_gemini(sde_prompt)
        
        # Step 2: Lead Software Developer (Review)

        
        lead_prompt = f"""
You are a Lead Software Developer.
Review the draft pseudo code for completeness and conciseness on each of the review criteria mentioned below.

Methodology & Technical Approach:

{methodology}

Draft Pseudo Code:
{draft_pseudo_code}

Review Criteria:
1. **Minimum Tables**: Does it identify the MINIMUM set of tables and columns required to build the Master Dataset for the defined KPIs?
2. **Step Count**: Are there 15 steps or fewer? If more, identify steps to consolidate.
3. **Completeness**: Does it cover all essential operations from the technical approach?
4. **Consolidation**: Can any adjacent steps be combined without losing clarity?
5. **Clarity**: Is each step clear and under 25 words?
6. **KPI Calculation**: Does it cover the calculation of all KPIs defined in the technical approach?
7. **Logic**: Does the sequence make sense and clear to understand for a non-technical person?
8. **Each step should be in a new line and visually separated from other steps.
9. **Prioritize tables with Rank 1.
10. **Explicitly mention table names and respective column names in the steps.
11.**Accuracy**: Is the pseudo code include relevant columns and tables suggested in the context and enables calculation of all KPIs calculated?
12.**Final Step**: Does the final step save to 'master_dataset.csv'?

Task:
- If the draft has 15 steps and meets all criteria, say "APPROVED".
- If it has >15 steps, suggest which steps to consolidate.
- If improvements needed, list them clearly and concisely.
"""
        feedback = call_gemini(lead_prompt)
        
        if "APPROVED" in feedback.upper() and len(feedback) < 50:
             return draft_pseudo_code

        # Step 3: Principal Software Engineer (Finalize)

        
        principal_prompt = f"""
You are a Principal Software Engineer.
Finalize the pseudo code by addressing feedback and ensuring maximum efficiency.

Draft Pseudo Code:
{draft_pseudo_code}

Lead Developer Feedback:
{feedback}

Methodology & Technical Approach:
{methodology}


CRITICAL REQUIREMENTS:
1. **Minimum Tables**: Identify the MINIMUM set of tables and columns required to build the Master Dataset for the defined KPIs.
2. **Maximum 10 Steps**: You MUST limit the pseudo code to 10 steps or fewer.
3. **Consolidate Operations**: Combine related operations (e.g., "Load data and standardize columns" instead of two separate steps).
4. **Strict Grounding**: You MUST use the tables and columns specified in the MASTER METADATA TABLE.
5. **Natural Language**: No code syntax. Use plain English descriptions.
6. Each step should be in a new line and visually separated from other steps.
7. Explicitly mention table names and respective column names in the steps.
8. **Concise**: Each step max 20-25 words.
9. **Final Step**: Must save to 'master_dataset.csv'.

Output Format:
Step 1: [Load col1, col2, ... from table_name 1 ...]
Step 2: [Consolidated action description]
...
Step N (≤10): Save the final processed dataframe to 'master_dataset.csv'.

Remember: MAXIMUM 10 STEPS TOTAL.
"""
        return call_gemini(principal_prompt)
    
    def refine(self, methodology, previous_pseudo_code, feedback):
        """
        Refine pseudo code based on user feedback.
        """
        prompt = f"""
You are an expert Software Development Engineer and Python Specialist.

Original Methodology & Technical Approach:
{methodology}

Current Pseudo Code:
{previous_pseudo_code}

User Feedback:
"{feedback}"

Task:
Refine the pseudo code to incorporate the user's feedback.

CRITICAL REQUIREMENTS:
1. **Minimum Tables**: Identify the MINIMUM set of tables and columns required to build the Master Dataset for the defined KPIs.
2. **Maximum 15 Steps**: You MUST limit the pseudo code to 15 steps or fewer.
3. **Consolidate Operations**: Combine related operations (e.g., "Load data and standardize columns" instead of two separate steps).
4. **High-Level Only**: Focus on major transformations, and always include table names and column names in each and every step.
5. **Natural Language**: No code syntax. Use plain English descriptions.
6. Each step should be in a new line and visually separated from other steps.
7. Prioritize tables with Rank 1.
8. Explicitly mention table names and respective column names in the steps.
9. **Concise**: Each step max 20-25 words.
10. **Final Step**: Must save to 'master_dataset.csv'.

Return the updated PSEUDO CODE with 15 steps.
"""
        

        return call_gemini(prompt)

    @trace_tool("Generate with Validation")
    def generate_with_validation(self, hypothesis, methodology, kpi_candidates, feedback=""):
        """
        Generates pseudocode with a Metadata-First flow.
    
        1. Identifies Master Metadata Table using GraphRAG candidates.
        2. Validates and retries Metadata Identification if needed.
        3. Generates Pseudocode using the IDENTIFIED Metadata as grounding.
        """
        from rag_retriever import RAGRetriever
        retriever = RAGRetriever()
        
        # 1. Phase 4.0: Identify Master Metadata Table (Grounding)
        master_metadata = retriever.identify_required_metadata(hypothesis, methodology, kpi_candidates)
        
        # Convert to markdown for the prompt context
        def to_markdown(ctx_list):
            if not ctx_list: return "No metadata identified."
            val = "| S.No | Table | KPIs | Columns | Reasoning |\n|---|---|---|---|---|\n"
            for row in ctx_list:
                if len(row) >= 5:
                    val += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
            return val
        
        metadata_md = to_markdown(master_metadata)

        
        # 2. Phase 4.1: Generate Logic using the Master Metadata as strict grounding
        pseudocode = self.generate(methodology, metadata_md, feedback=feedback)
        
        return pseudocode, master_metadata


    def _extract_context_list(self, pseudocode, candidates):
        prompt = f"""
        Extract the metadata usage from this pseudocode into a structured JSON list.
        
        Pseudocode:
        Pseudo code: {pseudocode}

        Available table schemas:
        {candidates}

        
        Task:
        Return a JSON nested list:
        [
            ["S.No", "Table", "KPIs", "Columns", "Reasoning"],
            ...
        ]
        
        Constraints:
        - Columns must be comma separated string.
        - JSON only.
        - 'S.No': Serial Number
        - 'Table': Table Name
        - 'KPIs': The specific KPI(s) this table contributes to.
        - 'Columns': All the columns (comma separated) pricked from the respective table. Make sure they are available in the table.
        - 'Reasoning': Reasoning on how the columns and tables are associated with the KPI.
    """

        result =call_gemini(prompt)

        return result
    
    def generate_for_kpi(self, kpi_name, methodology, data_context, granularity_analysis):
        """
        Generate pseudocode focused on creating a master dataset for a SINGLE KPI.
        """

        
        # Extract granularity info for this KPI
        granularity_info = "unknown"
        for group in granularity_analysis.get("mergeable_groups", []):
            if kpi_name in group.get("kpis", []):
                granularity_info = group.get("granularity", "unknown")
                break
        
        prompt = f"""
You are an expert Software Development Engineer creating a data pipeline.

Target KPI: {kpi_name}
Required Granularity: {granularity_info}

Methodology:
{methodology}

Available Tables and Columns (Master Metadata):
{data_context}

Task:
Create CONCISE pseudocode steps to build a master dataset SPECIFICALLY for calculating "{kpi_name}".

CRITICAL REQUIREMENTS:
1. **Single KPI Focus**: This dataset is ONLY for "{kpi_name}". Don't include data for other KPIs unless shared.
2. **Strict Grounding**: Use ONLY the tables/columns from the Master Metadata.
3. **Correct Granularity**: Ensure the final dataset is at "{granularity_info}" level.
4. **Maximum 10 Steps**: Keep it concise.
5. **Final Step**: Save to 'master_dataset_{kpi_name.lower().replace(' ', '_')}.csv'.

Output Format:
Step 1: Load columns [col1, col2] from [table_name]
Step 2: [Transformation/action]
...
Step N: Save final dataframe to 'master_dataset_{kpi_name.lower().replace(' ', '_')}.csv'

Return ONLY the numbered steps.
"""
        
        pseudocode = call_gemini(prompt)

        return pseudocode

# --- Convenience functions (as requested) ---
def generate_pseudo_code(methodology, data_context, mode='master'):
    """Generate pseudo code from methodology."""
    agent = PseudocodeAgent()
    return agent.generate(methodology, data_context, mode=mode)

def refine_pseudo_code(methodology, previous_pseudo_code, feedback):
    """Refine pseudo code based on feedback."""
    agent = PseudocodeAgent()
    return agent.refine(methodology, previous_pseudo_code, feedback)

