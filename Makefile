# End-to-end pipeline. `make all` reproduces every artifact from the raw CSVs in data/raw/.
PY ?= python
export PYTHONPATH := src:.
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: all setup ingest features audit train evaluate notebook predict serve docker-build docker-up test clean

all: ingest features audit train evaluate

setup:
	$(PY) -m pip install -r requirements.txt

ingest:         ## raw CSV -> validated parquet
	$(PY) -m thinfile.ingest

features: ## parquet -> features.parquet (sets A/B/C)
	$(PY) -m thinfile.features.build

audit:          ## leakage audit: can sets A/B tell who has a bureau file?
	$(PY) -m thinfile.audit

train:          ## CV + refit: 3 feature sets x {scorecard, lightgbm}
	$(PY) -m thinfile.train

evaluate:       ## test-set metrics, uplift tests, fairness, figures -> reports/
	$(PY) -m thinfile.evaluate

notebook:       ## run the walkthrough notebook (~12 min, ~3.5 GB RAM) and export an HTML copy
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 capstone_notebook.ipynb
	cd notebooks && jupyter nbconvert --to html capstone_notebook.ipynb

monitor:        ## PSI drift check on a simulated thin-file-heavy batch
	$(PY) -m thinfile.monitor --simulate

predict:        ## example batch scoring from the feature table
	$(PY) -m thinfile.predict --ids 100002 100003 100004

serve:          ## run the API locally on :8000
	uvicorn api.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d && sleep 5 && curl -s localhost:8000/health

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed/*.parquet artifacts/predictions artifacts/models mlflow.db reports/figures/*.png
