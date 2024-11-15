test:
	poetry run pytest --cov=gptme_rag

# run linting, typechecking, and tests
check:
	pre-commit run --all-files
