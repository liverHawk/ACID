uv run evaluate_model.py \
    --model_path results/trained_model_20251215_160531.pkl \
    --categories_path results/trained_model_20251215_160531.categories \
    --rf_model_path results/rf_model_20251215_160531.pkl \
    --dataset_path dataset/CICIDS2017_improved \
    --results_dir results \
    --train_test_split 0.7 \
    --random_seed 42

uv run visualize_adaptive_clustering.py \
    --model_path results/trained_model_20251215_160531.pkl \
    --categories_path results/trained_model_20251215_160531.categories \
    --dataset_path dataset/CICIDS2017_improved \
    --results_dir results \
    --n_samples 5000