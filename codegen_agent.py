import sys
import os
from state import GraphState
from observability import trace_node, trace_tool

# Add Data_Dictionary to path
if os.path.join(os.getcwd(), 'Data_Dictionary') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'Data_Dictionary'))
from gemini_client import call_gemini

"""
### Concept: Code Synthesis & Context Injection
The **Code Generation Agent** is a specialized coder. Its primary role is **Code Synthesis**—the process of turning abstract logic (Pseudocode) into executable instructions (Python).

A critical part of this is **Context Injection**. The agent isn't just writing generic Python; it's writing code that "knows" about your specific environment.

#### Key Mechanics:
1. **Metadata Injection**: We pass information about the actual CSV files (table names, column headers) directly into the prompt.
2. **Logic Steering**: We use the validated Pseudocode as a strict guide, ensuring the code follows the agreed-upon plan.
3. **Standardization**: The agent is instructed to use specific "guardrail" code (like `df.columns.str.strip()`) to ensure the resulting script is robust against common data issues.
"""

def clean_code_artifacts(code):
    """
    Remove all markdown and other artifacts from generated code.
    """
    if not code:
        return ""
    
    # Remove markdown code blocks
    lines = code.split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip markdown fence lines
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # Keep all other lines
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Additional cleanup - remove any stray backticks
    result = result.replace('```python', '').replace('```', '')
    
    return result

class CodeGenerationAgent:
    """Agent responsible for generating SQL and Python code using specialized roles."""
    
    @trace_node("CodeGen")
    def run(self, state: GraphState):
        """Generates per-KPI Python code artifact for LangGraph."""
        methodology = state["methodology"]
        metadata = state["metadata_context"]
        pseudocode_dict = state.get("pseudocode", {})
        if not isinstance(pseudocode_dict, dict):
            pseudocode_dict = {}
            
        existing_code_dict = state.get("python_code", {})
        if not isinstance(existing_code_dict, dict):
            existing_code_dict = {}
        current_kpi = state.get("current_kpi", "")
        
        if not current_kpi:
            return {
                "python_code": {},
                "current_step": "code_generated",
                "messages": [{"role": "assistant", "content": "Error: No KPI specified for code generation."}]
            }
        
        # Get pseudocode for this specific KPI
        kpi_pseudocode = pseudocode_dict.get(current_kpi, "")
        
        if not kpi_pseudocode:
            return {
                "python_code": existing_code_dict,
                "current_step": "code_generated",
                "messages": [{"role": "assistant", "content": f"Error: No pseudocode found for {current_kpi}."}]
            }
        
        # Check if refinement
        if existing_code_dict.get(current_kpi) and state.get("current_step") == "refine_codegen":
            error_message = state.get("latest_feedback") or (state["messages"][-1]["content"] if state["messages"] else "Resolve unknown issues.")
            python_code = self.refine_python(existing_code_dict[current_kpi], error_message, methodology, kpi_pseudocode)
        else:
            # Generate KPI-specific Python code
            python_code = self.generate_python_for_kpi(
                kpi_name=current_kpi,
                methodology=methodology,
                metadata=metadata,
                pseudocode=kpi_pseudocode
            )
        
        # Update dict
        updated_code_dict = existing_code_dict.copy()
        updated_code_dict[current_kpi] = python_code
        
        return {
            "python_code": updated_code_dict,
            "current_step": "code_generated",
            "messages": [{
                "role": "assistant",
                "content": f"✅ Python code generated for **{current_kpi}**"
            }]
        }

    def generate_sql(self, methodology, data_context, pseudo_code=None):
        """
        Generate SQL code based on methodology and optional pseudo code.
        """
        pseudo_code_context = f"\nApproved Pseudo Code:\n{pseudo_code}\n" if pseudo_code else ""
        
        prompt = f"""
You are an expert SQL Developer.

Context:
I have the following datasets available (CSV files can be treated as tables):
{data_context}

Methodology:
{methodology}
{pseudo_code_context}

Task:
Generate a single, executable SQL query (or a sequence of queries) to implement the data preparation and analysis steps described in the methodology.

CRITICAL REQUIREMENT - MASTER DATASET:
1. You MUST first create a "Master Dataset" (Semantic Layer) using a CTE (Common Table Expression).
2. This Master Dataset should aggregate the raw data to the appropriate grain as specified in the methodology.
3. All subsequent analysis and metrics calculation MUST be performed by querying this Master Dataset.
4. Do NOT query the raw tables repeatedly. Build the semantic layer once, then use it.

Structure:
```sql
WITH MasterDataset AS (
    -- Join tables, filter events, and aggregate as needed
    SELECT ...
),
Analysis AS (
    -- Perform hypothesis testing on MasterDataset
    SELECT ... FROM MasterDataset ...
)
SELECT * FROM Analysis;
```

Assume CSV files are loaded into tables with the same names.
Use generic SQL syntax. Add comments explaining each step.
"""

        return call_gemini(prompt)
    
    @trace_tool("Generate Python")
    def generate_python(self, methodology, data_context, pseudo_code=None, mode='master'):
        """
        Generate Python code based on methodology and pseudo code using Lead Developer persona.
        """
        pseudo_code_context = f"\nApproved Pseudo Code:\n{pseudo_code}\n" if pseudo_code else ""
        
        if mode == 'master':
            persona = "Lead Software Developer Expert in Python"
            task_specific_instructions = """
Task:
Generate a Python script to CREATE THE MASTER DATASET (Semantic Layer) implementing the provided Pseudo Code.
Focus on data loading, cleaning, joining, and aggregation.
Also include the final analysis/visualization to conclude the hypothesis test.

Requirements:
1. **Load Data**: Load necessary CSV files from 'Data_Dictionary/Datalake/'.
2. **Standardize**: Strip whitespace from column names immediately after loading.
3. **Filter**: Apply filters as specified.
4. **Aggregate**: Create the master dataset.
5. **Save**: Save final dataframe to 'master_dataset.csv' (CRITICAL: THIS FILE MUST BE CREATED).

Don't include any dummy data creation
"""
        else: # mode == 'analysis'
            persona = "Expert Python Analyst"
            task_specific_instructions = """
Task:
Generate a Python script to PERFORM ANALYSIS on the Master Dataset and CREATE A DASHBOARD.
Assume 'master_dataset.csv' exists in the current directory.

Requirements:
1. **Load Master**: Load 'master_dataset.csv'.
2. **KPI Calculation**: Calculate high-level KPIs.
3. **Analysis**: Perform statistical tests.
4. **Visualization**: Create Plotly/Seaborn charts.

Don't include any dummy data creation
"""

        prompt = f"""
You are a {persona}.

Context:
I have the following datasets available (CSV files):
{data_context}

Methodology:
{methodology}
{pseudo_code_context}

{task_specific_instructions}

Requirements (General):
- Use pandas for manipulation, seaborn/matplotlib for vis.
- Add comments explaining each step.
- Include proper error handling.
- Print intermediate shapes and results.

CRITICAL - DATA STANDARDIZATION (MANDATORY):
1. Immediately after loading ANY dataframe, you MUST standardize column names to remove whitespace: `df.columns = df.columns.str.strip()`.

CRITICAL - EMPTY DATA CHECKS:
1. After EVERY major transformation (filtering, joining), check if the result is empty and print a warning.

CRITICAL: Return ONLY raw Python code. NO markdown formatting. NO code blocks.
"""

        raw_code = call_gemini(prompt)
        return clean_code_artifacts(raw_code)
    
    @trace_tool("Refine Python")
    def refine_python(self, original_code, error_message, methodology=None, pseudo_code=None, mode='master'):
        """
        Refine Python code based on error message using a Principal Software Engineer persona.
        """
        methodology_context = f"\nOriginal Methodology:\n{methodology}\n" if methodology else ""
        pseudo_code_context = f"\nApproved Pseudo Code:\n{pseudo_code}\n" if pseudo_code else ""
        
        persona = "Principal Software Developer Python"
        
        prompt = f"""
You are a {persona}.
Your task is to fix the Python code that failed execution.

Context:
{methodology_context}
{pseudo_code_context}

Generated Python Code that Failed:
{original_code}

Validation Error / Traceback:
{error_message}

Task:
1. Analyze the error and fix it.
2. Ensure columns are stripped: `df.columns = df.columns.str.strip()`.
3. Fix the code to resolve the error efficiently.
4. Add debug prints.
5. Don't include any dummy data creation

Return ONLY the corrected Python code without any markdown formatting.
"""

        raw_code = call_gemini(prompt)
        return clean_code_artifacts(raw_code)
    
    @trace_tool("Generate KPI Python")
    def generate_python_for_kpi(self, kpi_name, methodology, metadata, pseudocode):
        """
        Generate Python code focused on creating a master dataset for a SINGLE KPI.
        """

        
        # Sanitize KPI name for filename
        kpi_filename = kpi_name.lower().replace(' ', '_').replace('/', '_')
        
        persona = "Lead Software Developer Python"
        
        prompt = f"""
You are a {persona} creating a data pipeline script.

Target KPI: {kpi_name}
Output File: master_dataset_{kpi_filename}.csv

Methodology:
{methodology}

Pseudocode (Approved Logic):
{pseudocode}

Available Tables and Columns (Master Metadata):
{metadata}

Task:
Generate a complete, executable Python script that:
1. Creates a master dataset SPECIFICALLY for calculating "{kpi_name}"
2. Follows the pseudocode EXACTLY
3. Uses ONLY the tables/columns from Master Metadata
4. Saves final dataset to 'master_dataset_{kpi_filename}.csv'

CRITICAL REQUIREMENTS:
1. **Imports**: `import pandas as pd`, `import os`
2. **Data Loading**: Read CSVs from `./Data_Dictionary/Datalake/` directory
3. **Column Stripping**: `df.columns = df.columns.str.strip()` after each load
4. **Grounding**: Use ONLY specified tables and columns
5. **Final Step**: 
```python
master_df.to_csv('master_dataset_{kpi_filename}.csv', index=False)
print(f"Master dataset saved to: master_dataset_{kpi_filename}.csv")
print(f"Shape: {{master_df.shape}}")
print(master_df.head())
```
6. **No Dummy Data**: Don't create dummy data
7. **Single KPI**: This is ONLY for "{kpi_name}"

Return ONLY the Python code without markdown formatting.
"""
        
        raw_code = call_gemini(prompt)
        cleaned_code = clean_code_artifacts(raw_code)

        return cleaned_code

# --- Convenience functions (as requested) ---
def generate_sql_code(methodology, data_context, pseudo_code=None):
    """Generate SQL code from methodology."""
    agent = CodeGenerationAgent()
    return agent.generate_sql(methodology, data_context, pseudo_code)

def generate_python_code(methodology, data_context, pseudo_code=None, mode='master'):
    """Generate Python code from methodology."""
    agent = CodeGenerationAgent()
    return agent.generate_python(methodology, data_context, pseudo_code, mode)

def refine_python_code(original_code, error_message, methodology=None, pseudo_code=None, mode='master'):
    """Refine Python code based on error."""
    agent = CodeGenerationAgent()
    return agent.refine_python(original_code, error_message, methodology, pseudo_code, mode)
