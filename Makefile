# Run the same checks as CI: `make check`.
PYTHON ?= python
export PYTHONPATH := src

.PHONY: install check lint test data train evaluate

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src scripts tests --select E4,E7,E9,F

test:
	$(PYTHON) -m unittest discover -s tests -v

check: lint test
	$(PYTHON) ci/verify_evidence.py

data:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train_nbeats.py --frequency Hourly --steps 4000
	$(PYTHON) scripts/train_nbeats.py --frequency Weekly --steps 2000
	$(PYTHON) scripts/train_nbeats.py --frequency Daily --steps 3000

evaluate:
	for f in Hourly Weekly Daily; do $(PYTHON) scripts/evaluate.py --frequency $$f; done
