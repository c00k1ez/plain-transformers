TEST_DATA_DIR := ./tests/test_data
TEST_TEXT_FILE := $(TEST_DATA_DIR)/lorem_ipsum.txt
TEST_YTTM_MODEL := $(TEST_DATA_DIR)/test_yttm_tokenizer.model

tests:
	yttm bpe --data $(TEST_TEXT_FILE) --model $(TEST_YTTM_MODEL) --vocab_size 200
	pytest

code_checks:
	black . --config=pyproject.toml --check
	flake8 . --config=setup.cfg
	isort . -sp=setup.cfg --recursive -s tests/ --check-only
