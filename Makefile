#!/bin/bash

.PHONY = clean-pyc clean-build setup lint test

help:
	@echo "	clean-pyc"
	@echo "		Remove python artifacts."
	@echo "	clean-build"
	@echo "		Remove build artifacts."
	@echo "	setup"
	@echo "		Install necessary dependencies."
	@echo "	lint"
	@echo "		Check style with flake8."
	@echo "	test"
	@echo "		Run unittests"

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

setup:
	python3 -m pip install --no-cache-dir --user -r requirements.txt
	python3 -m pip install --user -e .

lint:
	flake8 --exclude=.tox

test:
	python3 -m unittest tests
