#!/bin/bash

# Minimal makefile for Sphinx documentation
# and more!

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
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

.PHONY: help Makefile clean-pyc clean-build setup lint test

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

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