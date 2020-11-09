help:
	@echo "Commands:"
	@echo "install        : installs requirements."
	@echo "format         : runs flake8 style tests."
	@echo "clean          : cleans all unecessary files."
	@echo "check          : runs test, flake and clean."

install:
	pip install -r requirements.txt

run:
	bundle exec jekyll serve --livereload

clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	rm -f .coverage

generate:
	python generate.py