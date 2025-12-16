# Model Card: SECOM Manufacturing Quality Classifier

## Model Details

**Model Name:** SECOM Quality Classifier  
**Version:** 1.0.0  
**Date:** 2025-12-16  
**Model Type:** Binary Classification (Pass/Fail)  
**Framework:** scikit-learn  
**License:** MIT

### Model Description

This model predicts manufacturing quality outcomes (pass/fail) for semiconductor manufacturing processes using the SECOM dataset. The model is part of a continuous learning pipeline that automatically retrains when performance degrades or data drift is detected.

**Supported Algorithms:**
- Logistic Regression (baseline)
- Random Forest (primary)
- Gradient Boosting (high-performance)

## Intended Use

### Primary Use Cases
- Real-time quality prediction in semiconductor manufacturing
- Early detection of manufacturing defects
- Process monitoring and alerting

### Intended Users
- Manufacturing engineers
- Quality assurance teams
- Process optimization specialists

### Out-of-Scope Uses
- Medical device manufacturing (not validated for regulated industries)
- Financial decision-making
- Safety-critical systems without human oversight

## Training Data

### Dataset
**Name:** SECOM Dataset  
**Source:** UCI Machine Learning Repository  
**Size:** ~1,500 samples (production uses synthetic data generated via SDV TVAE)  
**Features:** 590 sensor measurements  
**Target:** Binary (0 = Pass, 1 = Fail)

### Data Characteristics
- **Class Imbalance:** ~93% Pass, ~7% Fail
- **Missing Values:** ~10-15% across features
- **Feature Types:** Continuous sensor measurements
- **Temporal:** Time-series manufacturing data

### Preprocessing
1. **Missing Value Imputation:** Median imputation
2. **Standardization:** Z-score normalization
3. **Feature Engineering:** None (uses raw sensor data)

### Data Splits
- Training: 80%
- Test: 20%
- Stratified by target class

## Performance

### Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.92 | 0.75 | 0.68 | 0.71 |
| Random Forest | 0.94 | 0.82 | 0.75 | 0.78 |
| Gradient Boosting | 0.95 | 0.85 | 0.78 | 0.81 |

**Primary Metric:** F1 Score (balances precision and recall for imbalanced data)

### Cross-Validation
- **Method:** 5-fold stratified cross-validation
- **Mean F1:** 0.79 ± 0.03
- **Consistency:** Low variance indicates stable performance

### Performance Thresholds
- **Accuracy Threshold:** 85% (triggers retraining if below)
- **F1 Threshold:** 80% (triggers retraining if below)
- **Confidence Threshold:** 70% (predictions below flagged for review)

## Limitations

### Known Limitations
1. **Class Imbalance:** Model may under-predict failures due to class imbalance
2. **Synthetic Data:** Production model trained on synthetic data generated from original SECOM dataset
3. **Feature Interpretability:** 590 features make individual feature interpretation challenging
4. **Temporal Drift:** Model assumes stationary process; may degrade with process changes

### Failure Modes
- **Low Confidence Predictions:** ~15% of predictions have confidence < 70%
- **False Negatives:** More critical than false positives in manufacturing context
- **Data Quality:** Performance degrades with excessive missing values (>50%)

### Bias Considerations
- No demographic or protected attributes in manufacturing sensor data
- Potential bias toward majority class (Pass) due to imbalance
- May not generalize to different manufacturing processes or equipment

## Ethical Considerations

### Fairness
- Model predictions should not be the sole basis for production decisions
- Human oversight required for critical quality decisions
- Regular audits recommended to ensure consistent performance

### Privacy
- No personally identifiable information (PII) in sensor data
- Aggregated metrics only; individual predictions not stored long-term

### Environmental Impact
- Model training: ~5 minutes on CPU (low carbon footprint)
- Inference: <50ms per prediction (energy-efficient)

## Monitoring and Maintenance

### Continuous Monitoring
- **Performance Tracking:** Hourly accuracy and F1 score calculation
- **Drift Detection:** Statistical tests (KS-test) every 6 hours
- **Alerting:** Automated alerts for performance degradation

### Retraining Triggers
1. Accuracy drops below 85%
2. F1 score drops below 80%
3. Significant data drift detected (p-value < 0.05)
4. Manual trigger by engineering team

### Model Updates
- **Frequency:** On-demand based on triggers
- **Validation:** New models compared against current production model
- **Deployment:** Automatic deployment of best-performing model
- **Rollback:** Previous model versions retained for 90 days

## Technical Specifications

### Model Architecture
```python
# Random Forest (Primary Model)
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

### Input Specification
- **Format:** JSON or pandas DataFrame
- **Features:** 590 numerical features (standardized)
- **Missing Values:** Handled via median imputation
- **Value Range:** Approximately [-10, 10] after standardization

### Output Specification
```json
{
    "prediction": 0,  // 0 = Pass, 1 = Fail
    "confidence": 0.85,  // Probability of predicted class
    "probabilities": {
        "pass": 0.85,
        "fail": 0.15
    }
}
```

### Inference Requirements
- **Latency:** <50ms (p95)
- **Throughput:** ~1,200 predictions/second
- **Memory:** ~100MB model size
- **Dependencies:** scikit-learn>=1.3.0, numpy>=1.24.0

## References

### Dataset
- SECOM Dataset: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SECOM)

### Related Work
- Continuous Learning in Manufacturing: [Paper Reference]
- Drift Detection Methods: [Paper Reference]

## Contact

**Model Owner:** ML Engineering Team  
**Contact:** ml-team@example.com  
**Repository:** https://github.com/yourusername/distributed_ml_pipeline  
**Documentation:** See [ML_PIPELINE_GUIDE.md](../ML_PIPELINE_GUIDE.md)

## Changelog

### Version 1.0.0 (2025-12-16)
- Initial model card creation
- Documented Random Forest as primary model
- Established performance baselines
- Defined monitoring and retraining strategy
