uv run evaluate_model.py \
    --model_path results/20251218_163559/trained_model_20251218_163559.pkl \
    --categories_path results/20251218_163559/trained_model_20251218_163559.categories \
    --rf_model_path results/20251218_163559/rf_model_20251218_163559.pkl \
    --dataset_path dataset/CICIDS2017_improved \
    --results_dir results/evaluate \
    --train_test_split 0.7 \
    --random_seed 42

uv run train_clustering.py \
  --config dataset_config.yaml \
  --dataset_path dataset/CICIDS2017_improved \
  --results_dir results/cluster \
  --logs_dir logs \
  --train_test_split 0.7 \
  --random_seed 42 \
  --learning_rate 1e-4 \
  --n_epochs 1

uv run run_known_unknown_clustering.py \
  --config dataset_config.yaml \
  --dataset_path dataset/CICIDS2017_improved \
  --results_dir results \
  --logs_dir logs \
  --train_test_split 0.7 \
  --random_seed 42 \
  --learning_rate 1e-4 \
  --n_epochs 1 \
  --early_stop_threshold 1.0

export FILE="results/adaptive_clustering_model" && uv run visualize_adaptive_clustering.py --model_path "$FILE".pkl --categories_path "$FILE".categories --dataset_path dataset/CICIDS2017_improved --results_dir results/ev