import os
import time
import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from openai import OpenAI

INPUT_CSV = "enterprise_software_dataset.csv"
PROMPT_A_FILE = "prompt_a_direct.txt"
PROMPT_B_FILE = "prompt_b_reasoning.txt"
OUTPUT_A = "results_prompt_a_openai.csv"
OUTPUT_B = "results_prompt_b_openai.csv"
MODEL_NAME = "gpt-4.1-mini"
SLEEP_BETWEEN_SUCCESS = 1.5

class EvidenceNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_audience: List[str]
    core_business_function: List[str]
    deployment_model: List[str]
    pricing_model: List[str]
    integration_depth: List[str]
    compliance_scope: List[str]

class ExtractedDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_audience: Optional[str]
    core_business_function: Optional[str]
    deployment_model: Optional[str]
    pricing_model: Optional[str]
    integration_depth: Optional[str]
    compliance_scope: Optional[str]

class OutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_notes: EvidenceNotes
    extracted_dimensions: ExtractedDimensions

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def init_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY as an environment variable.")
    return OpenAI(api_key=api_key)

def load_existing_results(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "sample_id" in df.columns:
            return df
    return pd.DataFrame()

def save_results(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")

def call_openai(client, prompt, raw_text):
    resp = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_text},
        ],
        text_format=OutputSchema,
    )
    return resp.output_parsed.model_dump()

def run_variant(client, df, prompt, variant_name, output_path):
    existing = load_existing_results(output_path)
    processed = set(existing["sample_id"].astype(str)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    print(f"\nRunning Variant {variant_name}...")
    print(f"Already completed: {len(processed)} rows")

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        product_name = row["product_name"]

        if sample_id in processed:
            print(f"Skipping [{sample_id}] - already saved")
            continue

        print(f"Processing [{sample_id}] - {product_name}...")
        try:
            result = call_openai(client, prompt, row["raw_text"])
            result["sample_id"] = sample_id
            result["product_name"] = product_name
            result["variant"] = variant_name
            rows.append(result)
            save_results(rows, output_path)
            processed.add(sample_id)
            time.sleep(SLEEP_BETWEEN_SUCCESS)
        except Exception as e:
            print(f"Error extracting from {sample_id}: {e}")
            continue

    return pd.DataFrame(rows)

def main():
    client = init_client()
    df = pd.read_csv(INPUT_CSV)

    prompt_a = load_text(PROMPT_A_FILE)
    prompt_b = load_text(PROMPT_B_FILE)

    results_a = run_variant(client, df, prompt_a, "A", OUTPUT_A)
    print(f"Variant A saved to {OUTPUT_A} ({len(results_a)} rows total)")

    results_b = run_variant(client, df, prompt_b, "B", OUTPUT_B)
    print(f"Variant B saved to {OUTPUT_B} ({len(results_b)} rows total)")

    print("\nDone. Compare results_prompt_a.csv and results_prompt_b.csv.")

if __name__ == "__main__":
    main()