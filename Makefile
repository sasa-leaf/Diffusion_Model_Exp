_isort_check:
	poetry run isort --check ./

_black_check:
	poetry run black --check ./

_mypy:
	poetry run mypy ./

lint:
	make -j _isort_check _black_check _mypy

_isort_apply:
	poetry run isort ./

_black_apply:
	poetry run black ./

fmt:
	make _isort_apply _black_apply

test:
	poetry run pytest