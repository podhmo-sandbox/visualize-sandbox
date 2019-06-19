test:
	python setup.py test

format:
#	pip install -e .[dev]
	black visualize_sandbox setup.py

lint:
#	pip install -e .[dev]
	flake8 visualize_sandbox --ignore W503,E203,E501

build:
#	pip install wheel
	python setup.py bdist_wheel

upload:
#	pip install twine
	twine check dist/visualize-sandbox-$(shell cat VERSION)*
	twine upload dist/visualize-sandbox-$(shell cat VERSION)*

.PHONY: test format lint build upload
