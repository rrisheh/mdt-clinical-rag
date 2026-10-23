# mdt-clinical-rag
# Specialty-Based Multi-Agent RAG for Discharge Diagnosis Prediction (MIMIC-IV)

Code for a controlled comparison of four LLM-based systems that predict the discharge
diagnoses of complex hospital admissions from MIMIC-IV, all using the same backbone
(MedGemma 27B, 4-bit GPTQ, served with vLLM):

1. **Model 1** – single LLM with the full patient record (baseline)
2. **Model 2** – single LLM with patient and clinical-guideline retrieval
3. **Model 3** – four specialist agents (Cardiology, Pulmonology, Nephrology,
   Infectious Diseases) and a moderator
4. **Model 3 + reasoning** – Model 3 with a one-line justification per diagnosis

Predictions are scored by three LLM judges (Llama 3.1 8B, Granite 4.1 8B, OLMo 3 7B)
using Recall@K, paired permutation tests, bootstrap confidence intervals and
Krippendorff's alpha.

## Data access

This repository contains **code only**. No MIMIC-IV data, patient cases, embeddings,
predictions or clinical guideline documents are included. Running the code requires
credentialed access to MIMIC-IV v3.1 through PhysioNet and Google BigQuery.
Patient and admission identifiers in the code have been replaced with placeholders
(`00000000`).

## Pipeline (run in this order)

| Notebook | Purpose |
|---|---|
| `1-Specialties-diseases scoping and cases selection` | Cohort, specialty tagging, CCSR disease families, coverage-aware sampling of 100 cases |
| `2-Patient cases extraction` | Builds each patient case and the held-out target file |
| `3-Patient_Chunking_Pipeline` | Section-aware adaptive semantic and table chunking |
| `4- Embedding_Patient_Cases` | Embeds patient chunks (BGE-base-en-v1.5) |
| `4-KB_Chunking_Pipeline` | Parses, chunks and filters the guideline knowledge base (Docling) |
| `Smoke-test_V2` | Local comparison of embedding models (BGE, Nomic, MedCPT, BM25) |
| `context-window analysis` | Input-length analysis defining the 60-case evaluation set |
| `5-Patient_Case_RAG` | Patient-chunk retrieval |
| `6-KB_RAG_Pipeline` | Guideline retrieval |
| `7-Shared_Evidence` | Precomputed evidence shared by Models 2 and 3 |
| `model1_pipeline` | Model 1 (also used for the MedGemma 4B vs 27B comparison) |
| `judging_backbone_4B_vs_27B` | Scoring and backbone selection |
| `model2_pipeline` | Model 2 |
| `model3_pipeline` | Model 3 |
| `model3r_pipeline` | Model 3 + reasoning |
| `judging_v2_hybrid` | Hybrid LLM-judge scoring of all models |
| `results_comparison` | Recall, statistical tests, judge agreement, explanation quality |

**Supporting analyses** (not pipeline steps): `Inspecting Radiology` (radiology section
headings used to design report cleaning) and `Inspecting_PE` (physical-examination
leakage checks).

## Environment

- Google Cloud Vertex AI Workbench, single NVIDIA L4 GPU (24 GB)
- Python with pandas, NumPy, scikit-learn, krippendorff, NLTK, Docling,
  sentence-transformers, openai (client for vLLM), google-cloud-bigquery
- vLLM for serving MedGemma 27B and the judge models (40,000-token context window)
- Fixed seeds: case sampling 42, bootstrap 1234; greedy decoding (temperature 0)
