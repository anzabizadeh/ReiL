init:
	python -m pip install --no-cache-dir -r requirements.txt

test:
	python -m unittest tests
