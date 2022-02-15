init: requirements.txt
	pip install -r requirements.txt

requirements.txt:
	pip install pipreqs
	pipreqs .

test:
	cd ccn && pytest *.py -vv

.PHONY: init test