import os
import json
import ast
import pandas as pd

file_map = {
    "OpenAI - Prompt A": "results_prompt_a_openai.csv",
    "OpenAI - Prompt B": "results_prompt_b_openai.csv",
    "Groq - Prompt A": "results_prompt_a.csv",
    "Groq - Prompt B": "results_prompt_b.csv"
}

# The standardized canonical categories we want to evaluate across formats
canonical_fields = ["target_audience", "core_functionality", "deployment_model", "pricing_structure", "integrations", "compliance_frameworks"]

# Cross-reference map to bridge OpenAI's exact terminology to our canonical metrics
openai_key_bridge = {
    "core_business_function": "core_functionality",
    "pricing_model": "pricing_structure",
    "integration_depth": "integrations",
    "compliance_scope": "compliance_frameworks"
}

legal_deployments = ["saas", "on-premise", "hybrid", "multi-cloud", "unknown"]
summary_records = []

print("Running Master Evaluation Aggregator (Aligned Schema Edition)...\n")

for config_name, file_name in file_map.items():
    if not os.path.exists(file_name):
        continue
        
    df = pd.read_csv(file_name)
    working_rows = []
    
    # Process OpenAI files
    if 'extracted_dimensions' in df.columns:
        for idx, row in df.iterrows():
            raw_val = row['extracted_dimensions']
            parsed_dict = {}
            if pd.notna(raw_val) and str(raw_val).strip() != "":
                try:
                    parsed_dict = ast.literal_eval(str(raw_val).strip())
                except Exception:
                    try:
                        parsed_dict = json.loads(str(raw_val).strip())
                    except Exception:
                        parsed_dict = {}
            
            # Map OpenAI variations to our standardized categories
            normalized_dict = {}
            for k, v in parsed_dict.items():
                target_key = openai_key_bridge.get(k, k)
                normalized_dict[target_key] = v
            working_rows.append(normalized_dict)
            
        working_df = pd.DataFrame(working_rows)
        for field in canonical_fields:
            if field not in working_df.columns: 
                working_df[field] = ""
    else:
        # Process Groq files
        working_df = df.copy()

    # Determine which metric fields are available for analysis
    available_fields = [f for f in canonical_fields if f in working_df.columns]
    if not available_fields:
        continue

    # Evaluate by row positions (Rows 0-4 are Handwritten, Rows 5+ are LLM-Generated)
    for text_type in ["Handwritten", "LLM-Generated"]:
        subset = working_df.iloc[0:5] if text_type == "Handwritten" else working_df.iloc[5:]
        if subset.empty:
            continue
            
        total_samples = len(subset)
        
        # 1. Missing Info Rate Metric
        unknown_count = 0
        for field in available_fields:
            unknown_count += subset[field].astype(str).str.lower().str.strip().replace(r'^\s*$', 'unknown', regex=True).str.contains('unknown|not specified|none mentioned|nan').sum()
        total_fields_tracked = total_samples * len(available_fields)
        missing_data_pct = (unknown_count / total_fields_tracked) * 100
        
        # 2. Text Verbosity Metric
        avg_word_count = subset[available_fields].astype(str).apply(lambda x: x.str.split().str.len()).mean().mean()
        
        # 3. Schema Deviation Metric (Checks if any valid enum term is present)
        schema_deviations = 0
        if "deployment_model" in working_df.columns:
            for val in subset["deployment_model"].astype(str):
                cleaned_val = val.strip().lower()
                # A deviation occurs if the field does not contain any of our expected keywords
                has_valid_keyword = any(enum_item in cleaned_val for enum_item in legal_deployments)
                if not has_valid_keyword and cleaned_val != "nan":
                    schema_deviations += 1
        schema_deviation_pct = (schema_deviations / total_samples) * 100
        
        summary_records.append({
            "Configuration": config_name,
            "Text Texture": text_type,
            "Sample Count": total_samples,
            "Omission Rate (%)": round(missing_data_pct, 1),
            "Avg Field Words": round(avg_word_count, 1),
            "Schema Deviation (%)": round(schema_deviation_pct, 1)
        })

metrics_df = pd.DataFrame(summary_records)
metrics_df.to_csv("final_academic_metrics.csv", index=False)

print("--- ALL CRITERIA COVERED: TRUE ALIGNED METRICS ---")
print(metrics_df.to_string(index=False))