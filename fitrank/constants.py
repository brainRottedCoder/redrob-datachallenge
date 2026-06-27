"""Shared constants used across scoring modules."""

from __future__ import annotations

# Company names that the JD explicitly flags as consulting/services-only.
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "genpact", "ibm consulting", "deloitte",
    "ey", "kpmg", "pwc", "mckinsey", "bain", "bcg",
}

# Titles that are research-only and should be penalized without production evidence.
RESEARCH_ONLY_TITLES = {
    "research scientist", "research engineer", "postdoc", "research fellow",
}

# PhD is a degree, not a title; check it as a standalone title prefix/suffix only.
PHD_TITLE_PATTERNS = ("phd researcher", "phd research", "phd engineer", "phd scientist")

# Titles that are CV/speech/robotics and should be penalized without NLP/IR evidence.
CV_SPEECH_ROBOTICS_TITLES = {
    "computer vision", "cv engineer", "vision engineer", "speech recognition",
    "speech engineer", "robotics engineer", "roboticist", "control engineer",
}

# Tier-1 Indian cities preferred by the JD.
INDIAN_TIER1 = {
    "pune", "noida", "bangalore", "bengaluru", "mumbai", "delhi", "gurgaon",
    "gurugram", "hyderabad", "chennai", "kolkata", "ahmedabad", "secunderabad",
    "faridabad", "ghaziabad", "navi mumbai",
}

# Deep ML/IR keywords used for reasoning extraction and other checks.
ML_KEYWORDS = [
    "LoRA", "QLoRA", "RLHF", "DPO", "SFT", "RAG", "PEFT", "fine-tuning",
    "vector search", "semantic search", "embedding", "sentence-transformer",
    "FAISS", "Milvus", "Weaviate", "Pinecone", "Qdrant", "PyTorch", "TensorFlow",
    "Hugging Face", "model deployment", "model serving", "inference serving",
    "MLflow", "Weights & Biases", "BentoML", "Triton", "hyperparameter",
    "distributed training", "LLaMA", "Mistral", "recommendation", "ranking",
    "NDCG", "MRR", "MAP", "A/B testing", "learning-to-rank", "XGBoost",
]
