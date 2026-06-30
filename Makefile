.PHONY: install test run validate deck precompute train download-model eval ablation

install:
	pip install -r requirements.txt

download-model:
	python -c "from fitrank.embedder import download_model; download_model()"

test:
	pytest tests/ -v

test-fast:
	pytest -m "not slow" -q

precompute:
	python scripts/precompute_embeddings.py --candidates data/candidates.jsonl --output-dir outputs/

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

validate:
	python validate_submission.py outputs/submission.csv

deck:
	python scripts/generate_deck_pdf.py

reproduce: install download-model precompute train run validate
