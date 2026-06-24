.PHONY: install test run validate

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out outputs/submission.csv

validate:
	python validate_submission.py outputs/submission.csv
