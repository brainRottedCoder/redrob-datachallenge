.PHONY: install install-llm test test-fast run validate deck precompute precompute-reasoning train download-model eval ablation sensitivity reproduce

install:
	pip install -r requirements.txt

install-llm:
	pip install -r requirements-llm.txt

download-model:
	python -c "from fitrank.embedder import download_model; download_model()"

test:
	pytest tests/ -v

test-fast:
	pytest -m "not slow" -q

precompute:
	python scripts/precompute_embeddings.py --candidates data/candidates.jsonl --output-dir outputs/

precompute-reasoning:
	python scripts/precompute_reasoning.py --candidates data/candidates.jsonl --jd data/job_description.txt --cache outputs/reasoning_cache.jsonl

train:
	python scripts/train_learned_ranker.py

run:
	python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv

run-calibrated:
	python rank.py --calibrate --out outputs/submission_calibrated.csv

eval:
	python rank.py --eval

ablation:
	python scripts/ablation_analysis.py

sensitivity:
	python scripts/run_sensitivity.py --candidates data/candidates.jsonl --jd data/job_description.txt --rebuild-ground-truth

validate:
	python validate_submission.py outputs/submission.csv

deck:
	python scripts/generate_deck_pdf.py

reproduce: install download-model precompute train run validate
