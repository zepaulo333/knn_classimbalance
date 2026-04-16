.PHONY: env test notebook clean

env:
	conda env create -f environment.yml

test:
	pytest tests/ --cov=src --cov-report=term-missing

notebook:
	jupyter notebook notebooks/

clean:
	rm -rf results/figures/* results/tables/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
