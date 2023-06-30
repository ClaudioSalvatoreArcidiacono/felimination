env-create:
	pip install --upgrade pip
	if ! test -d venv; \
	then \
		echo creating virtual environment; \
		pip install --upgrade virtualenv; \
		python -m venv venv; \
	fi

env-install:
	pip install --upgrade pip
	if test -s requirements.txt; \
	then \
		echo Installing requirements from requirements.txt; \
		pip install -r requirements.txt ; \
		pip install -e . --no-deps ; \
	else \
		echo Installing requirements from pyproject.toml; \
		pip install -e '.[dev]'; \
		pip freeze --exclude-editable > requirements.txt; \
	fi

env-update:
	pip install -e '.[dev]'
	pip freeze --exclude-editable > requirements.txt