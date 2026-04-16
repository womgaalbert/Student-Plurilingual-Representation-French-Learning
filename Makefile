.PHONY: setup preprocess train-h1 train-h2 train-h3 train-h4 train-all evaluate test mlflow-ui clean

setup:
	python -m venv .venv
	.venv\Scripts\pip install --upgrade pip
	.venv\Scripts\pip install -r requirements.txt

preprocess:
	python src/preprocess.py --config params.yaml

train-h1:
	python src/train.py --hypothesis H1 --config params.yaml

train-h2:
	python src/train.py --hypothesis H2 --config params.yaml

train-h3:
	python src/train.py --hypothesis H3 --config params.yaml

train-h4:
	python src/train.py --hypothesis H4 --config params.yaml

train-all:
	python src/pipeline.py --config params.yaml

evaluate:
	python src/evaluate.py --config params.yaml

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
