PYTHON ?= uv run
DATASET_PATH ?= dataset/CICIDS2017_improved
RESULTS_DIR ?= results
LOGS_DIR ?= logs
CONFIG ?= dataset_config.yaml
DEVICE ?= auto
EPOCHS ?= 1

.PHONY: help init venv install train_ids train_ids_cicids2017 train_cluster known_unknown eval eval_cluster visualize clean

help:
	@echo "Available targets:"
	@echo "  init                - install dependencies via uv"
	@echo "  train_ids           - train IDS (known-only) on CICIDS2017_improved"
	@echo "  train_ids_cicids2017 - train IDS using IDS_cicids2017.py (original script)"
	@echo "  train_cluster       - train adaptive clustering model"
	@echo "  known_unknown       - run known/unknown clustering experiment"
	@echo "  eval                - evaluate a saved model (set MODEL_TS)"
	@echo "  eval_cluster        - evaluate clustering model (set MODEL_PATH and CATEGORIES_PATH)"
	@echo "  visualize           - visualize adaptive clustering (set VIS_TS)"
	@echo "  clean               - remove cached artifacts (__pycache__, logs, results)"

init:
	uv sync

venv: init
	@echo "venv is managed by uv (see pyproject.toml)"

install: init

train_ids:
	$(PYTHON) run_ids_cicids2017_known.py \
		--config $(CONFIG) \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR) \
		--logs_dir $(LOGS_DIR) \
		--learning_rate 1e-4 \
		--n_epochs $(EPOCHS) \
		--early_stop_threshold 1.0 \
		--device $(DEVICE)

train_ids_cicids2017:
	$(PYTHON) IDS_cicids2017.py \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR) \
		--learning_rate 1e-4 \
		--n_epochs $(EPOCHS) \
		--early_stop_threshold 1.0 \
		--device $(DEVICE)

train_cluster:
	$(PYTHON) train_clustering.py \
		--config $(CONFIG) \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR)/cluster \
		--logs_dir $(LOGS_DIR) \
		--train_test_split 0.7 \
		--random_seed 42 \
		--learning_rate 1e-4 \
		--n_epochs $(EPOCHS) \
		--device $(DEVICE)

known_unknown:
	$(PYTHON) run_known_unknown_clustering.py \
		--config $(CONFIG) \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR) \
		--logs_dir $(LOGS_DIR) \
		--train_test_split 0.7 \
		--random_seed 42 \
		--learning_rate 1e-4 \
		--n_epochs $(EPOCHS) \
		--early_stop_threshold 1.0 \
		--device $(DEVICE)

# MODEL_TS: タイムスタンプ (例: 20251218_163559)
# 例: make eval MODEL_TS=20251218_163559

eval:
	@if [ -z "$(MODEL_TS)" ]; then \
		echo "ERROR: MODEL_TS is not set. Usage: make eval MODEL_TS=YYYYmmdd_HHMMSS"; \
		exit 1; \
	fi
	$(PYTHON) evaluate_model.py \
		--model_path $(RESULTS_DIR)/$(MODEL_TS)/trained_model_$(MODEL_TS).pkl \
		--categories_path $(RESULTS_DIR)/$(MODEL_TS)/trained_model_$(MODEL_TS).categories \
		--rf_model_path $(RESULTS_DIR)/$(MODEL_TS)/rf_model_$(MODEL_TS).pkl \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR)/evaluate \
		--train_test_split 0.7 \
		--random_seed 42

# MODEL_PATH: モデルファイルのパス (例: results/adaptive_clustering_model.pkl)
# CATEGORIES_PATH: カテゴリファイルのパス (例: results/adaptive_clustering_model.pkl.categories)
# 例: make eval_cluster MODEL_PATH=results/adaptive_clustering_model.pkl CATEGORIES_PATH=results/adaptive_clustering_model.pkl.categories

eval_cluster:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "ERROR: MODEL_PATH is not set. Usage: make eval_cluster MODEL_PATH=path/to/model.pkl CATEGORIES_PATH=path/to/categories"; \
		exit 1; \
	fi
	@if [ -z "$(CATEGORIES_PATH)" ]; then \
		echo "ERROR: CATEGORIES_PATH is not set. Usage: make eval_cluster MODEL_PATH=path/to/model.pkl CATEGORIES_PATH=path/to/categories"; \
		exit 1; \
	fi
	$(PYTHON) evaluate_clustering_model.py \
		--model_path $(MODEL_PATH) \
		--categories_path $(CATEGORIES_PATH) \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR) \
		--logs_dir $(LOGS_DIR) \
		--train_test_split 0.7 \
		--random_seed 42 \
		--device $(DEVICE)

# VIS_TS: タイムスタンプ (例: 20251218_212948)
# 例: make visualize VIS_TS=20251218_212948

visualize:
	@if [ -z "$(VIS_TS)" ]; then \
		echo "ERROR: VIS_TS is not set. Usage: make visualize VIS_TS=YYYYmmdd_HHMMSS"; \
		exit 1; \
	fi
	$(PYTHON) visualize_adaptive_clustering.py \
		--model_path $(RESULTS_DIR)/$(VIS_TS)/trained_model_$(VIS_TS).pkl \
		--categories_path $(RESULTS_DIR)/$(VIS_TS)/trained_model_$(VIS_TS).categories \
		--dataset_path $(DATASET_PATH) \
		--results_dir $(RESULTS_DIR) \
		--logs_dir $(LOGS_DIR) \
		--device $(DEVICE)

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
	rm -rf $(LOGS_DIR)/* $(RESULTS_DIR)/*
