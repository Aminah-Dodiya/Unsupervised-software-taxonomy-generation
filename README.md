# Unsupervised Software Taxonomy Generation: An Empirical Benchmark of Direct and Chain-of-Thought Prompting on Frontier LLMs

This repository contains the complete experimental design science research framework, python source code, and dataset pipelines for generating and evaluating unsupervised business-to-business (B2B) enterprise software and Software-as-a-Service (SaaS) product taxonomies. 

The core goal of this pipeline is to evaluate the trade-offs between prompting architectures, parameter weights, and incoming text textures (human-written vs. machine-generated synthetic text) under strict validation constraints.

## 🏛️ Repository Blueprint & Structure

The project environment is divided into clear functional directories to ensure modular execution and reproduction:

```text
├── prompts/
│ ├── prompt_a_direct_extraction.txt # Frozen System Instructions (Control Group)
│ └── prompt_b_chain_of_thought.txt # Frozen System Instructions (Experimental Group)
│
├── data/
│ ├── input_software_catalog.csv # 30 Base software profiles (5 human seeds, 25 synthetic)
│ ├── raw_output_openai_direct.csv # Raw extraction tables from gpt-4.1-mini
│ ├── raw_output_openai_cot.csv # Raw extraction tables from gpt-4.1-mini
│ ├── raw_output_groq_direct.csv # Raw extraction tables from llama-3.3-70b-versatile
│ └── raw_output_groq_cot.csv # Raw extraction tables from llama-3.3-70b-versatile
│
├── src/
│ ├── dataset_generator.py # Extrapolates 25 synthetic profiles from 5 human seeds
│ ├── pipeline_openai.py # Batch processes catalog rows via OpenAI endpoint
│ ├── pipeline_groq.py # Batch processes catalog rows via Groq instructor endpoint
│ └── evaluation_metrics_engine.py # Core statistical grading script
│
├── output/
│ ├── final_academic_metrics.csv # Master quantitative performance evaluation matrix
│ ├── fig2_performance_heatmap.png # Consolidated behavior heatmap matrix
│ ├── fig3_omission_rate_chart.png # Stratified omission rate percentage comparison
│ ├── fig4_field_verbosity_chart.png # Parsimony check tracking average field word counts
│ └── fig5_pareto_frontier_scatter.png # Multi-criteria optimization scatter plot
│
└── Taxonomy_Presentation_Dodiya_SS2026.pptx # Official academic presentation slide deck
```

## 📊 Core Empirical Findings

Our evaluation engine measured performance across three target metrics: Omission Rate (%), Average Field Words, and Schema Deviation (%). The experimental results proved a clear **Pareto Frontier**:
1. **Operational Database Winner:** Groq (`llama-3.3-70b-versatile`) + Prompt A achieved a perfect **0.0% omission rate** and a concise **5.8 words per field** on handwritten human text, satisfying Nickerson's criteria for parsimony.
2. **The Verbosity Leak Outlier:** OpenAI (`gpt-4.1-mini`) + Prompt B suffered a heavy length explosion averaging **14.4 words per field**, proving that compact models experience a reasoning bleed when forced to think out loud within a single prompt layer.
3. **Synthetic Data Texture Bias:** When processing pure machine-generated synthetic text arrays, formatting and mapping errors systematically spiked to **60.0% - 68.0%** across all configurations, highlighting that human linguistic noise is mandatory to keep LLM context tracking stable.

## 📚 Foundational Literature Pillars

This implementation bridges four core theoretical academic frameworks:
* **Nickerson et al. (2013):** Taxonomy Development Methodology for Information Systems.
* **Wei et al. (2022):** Chain-of-Thought Prompting Mechanisms.
* **Gao et al. (2023):** Schema Adherence and Pydantic Formatting Constraints.
* **Manning et al. (2024):** Semantic Compression Anomalies and Linguistic Ambiguity.