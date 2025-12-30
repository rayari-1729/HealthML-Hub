# Evaluation Standards for HealthML-Hub

## Overview

This document defines the evaluation standards and metrics used across all models in HealthML-Hub. These standards ensure consistent, rigorous evaluation of medical ML models.

## Why Accuracy Alone is Insufficient

### The Problem with Accuracy

In medical ML, accuracy can be misleading:
- **Class Imbalance**: Many medical datasets are imbalanced (e.g., 90% healthy, 10% disease)
- **Cost of Errors**: False negatives (missing disease) are often more costly than false positives
- **Clinical Context**: Different metrics matter for different use cases

### Example

A model that always predicts "no disease" would have 90% accuracy on an imbalanced dataset, but would be completely useless.

## Metrics by Disease Category

### Binary Classification Models

All binary classification models in HealthML-Hub report:

#### Primary Metrics

1. **Accuracy**: Overall correctness
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Use: Overall performance indicator

2. **Precision**: Of positive predictions, how many are correct
   - Formula: TP / (TP + FP)
   - Use: When false positives are costly

3. **Recall (Sensitivity)**: Of actual positives, how many are found
   - Formula: TP / (TP + FN)
   - Use: When false negatives are costly (most medical cases)

4. **F1 Score**: Harmonic mean of precision and recall
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)
   - Use: Balanced metric when both precision and recall matter

5. **ROC-AUC**: Area under the ROC curve
   - Range: 0.0 to 1.0
   - Use: Overall model discrimination ability
   - Interpretation:
     - 0.9-1.0: Excellent
     - 0.8-0.9: Good
     - 0.7-0.8: Fair
     - <0.7: Poor

#### Clinical Metrics

6. **Sensitivity (Recall)**: True positive rate
   - Critical for screening applications
   - Higher is better

7. **Specificity**: True negative rate
   - Formula: TN / (TN + FP)
   - Important for confirmatory tests

8. **Confusion Matrix**: Full breakdown
   - True Positives (TP)
   - True Negatives (TN)
   - False Positives (FP)
   - False Negatives (FN)

### Deep Learning Models

Deep learning models (e.g., brain_tumor) use the same metrics but may also report:
- **Training/Validation Loss**: For monitoring training
- **Learning Curves**: For detecting overfitting

## Confidence & Abstention Philosophy

### Confidence Calculation

All models calculate confidence as:
```python
confidence = 2 * abs(probability - 0.5)
```

This measures how far the prediction probability is from uncertainty (0.5).

### Confidence Thresholds

- **High Confidence** (≥0.8): Model is very certain
- **Medium Confidence** (0.7-0.8): Model is reasonably certain
- **Low Confidence** (<0.7): Model is uncertain, warning is generated

### Abstention Response

When confidence is low or prediction fails, models return:
```python
{
    "prediction": None,
    "probability": None,
    "confidence": 0.0,
    "warning": "Low confidence prediction or prediction failed"
}
```

This allows consuming systems to:
- Request additional information
- Escalate to human review
- Use alternative models
- Return "insufficient information" to user

## Evaluation Methodology

### Data Splitting

All models use:
- **Training Set**: 60-80% of data
- **Validation Set**: 10-20% of data (for hyperparameter tuning)
- **Test Set**: 10-20% of data (held out, only for final evaluation)

### Stratification

For imbalanced datasets:
- Stratified splits ensure class distribution is maintained
- Prevents bias in train/test splits

### Cross-Validation

For small datasets:
- K-fold cross-validation (typically k=5 or k=10)
- Provides more robust performance estimates

## Model-Specific Evaluation

### Diabetes Model

**Critical Metrics**: Sensitivity (recall) - missing diabetes is dangerous
**Target Performance**:
- Sensitivity: >0.85
- Specificity: >0.80
- ROC-AUC: >0.85

### Heart Disease Model

**Critical Metrics**: Sensitivity - early detection is crucial
**Target Performance**:
- Sensitivity: >0.90
- Specificity: >0.75
- ROC-AUC: >0.88

### Breast Cancer Model

**Critical Metrics**: Both sensitivity and specificity
**Target Performance**:
- Sensitivity: >0.95 (minimize false negatives)
- Specificity: >0.90 (minimize false positives)
- ROC-AUC: >0.95

### Brain Tumor Model

**Critical Metrics**: Sensitivity - missing tumors is critical
**Target Performance**:
- Sensitivity: >0.92
- Specificity: >0.88
- ROC-AUC: >0.90

## Evaluation Reporting

### Standard Report Format

All evaluation scripts generate reports with:
1. **Confusion Matrix**: Visual and numerical
2. **Primary Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
3. **Clinical Metrics**: Sensitivity, Specificity
4. **Per-Class Metrics**: For multi-class problems
5. **Confidence Distribution**: Histogram of confidence scores

### Registry Metadata

Each model's `artifacts/registry.json` contains:
```json
{
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "roc_auc": 0.90,
        "sensitivity": 0.88,
        "specificity": 0.83
    },
    "evaluation_date": "2024-01-01",
    "test_set_size": 200,
    "evaluation_notes": "Evaluated on held-out test set"
}
```

## Continuous Evaluation

### Model Monitoring

In production (future):
- Track prediction distributions
- Monitor confidence scores
- Alert on performance degradation
- A/B testing for model updates

### Benchmarking

Regular benchmarking against:
- Baseline models (e.g., logistic regression)
- Published results
- Clinical standards

## Limitations and Considerations

### Dataset Limitations

- **Synthetic Data**: Current models use synthetic data for demonstration
- **Population Bias**: Models may not generalize to all populations
- **Temporal Drift**: Medical knowledge and practices evolve

### Evaluation Limitations

- **Test Set Size**: Small test sets may not be representative
- **Distribution Shift**: Test data may differ from training data
- **Label Quality**: Ground truth labels may have errors

## Best Practices

### For Model Developers

1. **Report All Metrics**: Don't cherry-pick favorable metrics
2. **Document Limitations**: Be transparent about model weaknesses
3. **Use Appropriate Metrics**: Choose metrics based on clinical context
4. **Validate on Multiple Datasets**: Test generalization
5. **Report Confidence**: Always include confidence scores

### For Model Consumers

1. **Check Confidence**: Don't trust low-confidence predictions
2. **Consider Context**: Use appropriate metrics for your use case
3. **Validate Independently**: Test on your own data
4. **Monitor Performance**: Track model performance over time
5. **Understand Limitations**: Read model documentation carefully

## Future Improvements

### Planned Enhancements

1. **Calibrated Probabilities**: Better probability estimates
2. **Uncertainty Quantification**: Bayesian methods, ensemble uncertainty
3. **Fairness Metrics**: Ensure models work for all populations
4. **Explainability Metrics**: Quantify explanation quality
5. **Clinical Validation**: Real-world validation studies

## References

- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Medical ML Evaluation Best Practices](https://www.nature.com/articles/s41591-021-01614-0)
- [ROC Curve Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

## Conclusion

Rigorous evaluation is essential for medical ML models. HealthML-Hub follows these standards to ensure:
- **Transparency**: Clear reporting of all metrics
- **Reproducibility**: Consistent evaluation methodology
- **Clinical Relevance**: Metrics that matter for medical applications
- **Safety**: Confidence and abstention mechanisms

All models in HealthML-Hub are evaluated according to these standards, with results documented in model registries and evaluation reports.

