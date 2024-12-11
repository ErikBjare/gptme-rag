test:
	poetry run pytest --cov=gptme_rag --durations=5

# run linting, typechecking, and tests
check:
	pre-commit run --all-files

typecheck:
	pre-commit run mypy --all-files
