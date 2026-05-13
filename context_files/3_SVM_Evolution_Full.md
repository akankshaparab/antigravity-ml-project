# Phase 2: SVM Classification and Query Routing

## Performance Evolution

### Baseline (Academic Benchmark - ml_project)
Initial classification metrics showed linguistic overlap and a performance floor for 'Extra Hard' queries.

![Baseline Confusion Matrix](baseline_confusion_matrix.png)
![Baseline Metrics](baseline_metrics.png)

### Production (SAGE Platform - produc_vers)
After RBF kernel implementation and targeted index augmentation, the F1-score jumped to 0.81.

![Production Confusion Matrix](production_confusion_matrix.png)
![Production Metrics](production_metrics.png)
