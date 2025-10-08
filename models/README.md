# Models Directory

This directory contains trained models for the SECOM ML pipeline:

- `sdv_secom_raw.joblib` - Trained SDV (TVAE) model for synthetic data generation
- `preprocessing_pipeline.joblib` - Preprocessing pipeline (imputation + scaling)
- `best_model_*.joblib` - Trained classification models

## Training Models

To train the SDV model:
```bash
python data_generator/secom_raw_trainer.py
```

This will generate `sdv_secom_raw.joblib` which is used by the producer.

## Model Files

The preprocessing pipeline is created during the EDA process and saved here for use in production.
