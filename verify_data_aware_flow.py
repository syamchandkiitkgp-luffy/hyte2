
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from methodology_agent import MethodologyAgent

def test_data_aware_flow():
    print("\n[VERIFICATION] Starting Data-Aware Sequential Flow Test...")
    agent = MethodologyAgent()
    
    hypothesis = "Testing if 5G penetration leads to lower churn in metropolitan areas."
    
    # Mock initial_strategy_dict (Phase 3 output)
    initial_strategy_dict = {
        "approach": "1. Analyze 5G vs 4G usage patterns.\n2. Correlate with churn status.",
        "assumptions": "1. 5G availability is mapped to towers.",
        "kpis": {
            "5G Churn Rate": "Churn rate specifically for 5G users.",
            "5G Penetration": "Ratio of 5G to 4G users."
        },
        "visualizations": [
            {"Key": "Line Chart", "Value": "Trends over time."}
        ]
    }
    
    # Test Stage 2 Generation
    print("\n[TEST] Testing _get_data_strategy_manager_final...")
    result = agent.generate_data_aware_methodology(hypothesis, initial_strategy_dict)
    
    print("\n[RESULT] Data-Aware Methodology Output:")
    print(result.get("methodology"))
    
    print("\n[RESULT] Metadata Context Table:")
    print(result.get("metadata_context"))
    
    print("\n[RESULT] Pseudocode Preview:")
    print(result.get("pseudocode")[:200] + "...")
    
    # Validation checks
    methodology = result.get("methodology", "")
    assert "Approach" in methodology, "Missing Approach in final methodology"
    assert "KPIs" in methodology, "Missing KPIs in final methodology"
    assert "Visualizations" in methodology, "Missing Visualizations in final methodology"
    assert "Assumptions" in methodology, "Missing Assumptions in final methodology"
    
    # Check if data context is actually used (heuristic check)
    # Since we use RAG, it should find 'tower' or 'customer' or 'churn' related tables
    metadata_context = result.get("metadata_context", "")
    assert "|" in metadata_context, "Metadata context should be a markdown table"
    
    print("\n[VERIFICATION] SUCCESS: Data-Aware Sequential Flow is working correctly.")

if __name__ == "__main__":
    try:
        test_data_aware_flow()
    except Exception as e:
        print(f"\n[VERIFICATION] ❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
