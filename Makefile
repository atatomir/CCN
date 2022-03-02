init: requirements.txt
	pip install -r requirements.txt
	pip install .

requirements.txt:
	pip install pipreqs
	pipreqs .

test:
	cd ccn && pytest *.py -vv --durations=10

testx:
	cd ccn && pytest *.py -vv --durations=10 -x

.PHONY: init test
