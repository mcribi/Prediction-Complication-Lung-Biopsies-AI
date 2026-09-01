# Radiomic ML models inside discovered subgroups

Exploratory matched comparison between a full-cohort model evaluated inside each subgroup and a model trained only on subgroup members.

- Reference features: R11, 107 basic vessel radiomic features.
- Outer folds: reused exactly from R11.
- Labels: Hemorrhage and Pneumothorax; No complication derived as `(0, 0)`.
- Models: dummy, logistic regression, linear SVM, Gaussian NB, Random Forest, Extra Trees, LightGBM.
- Fold-local processing: median imputation, variance threshold 0.01, SelectKBest up to 20, standardization.
- Decisions: fixed 0.5 and thresholds selected by inner OOF F1.
- Primary comparison: specialized minus general tuned OOF F1 micro.

The subgroup rules and R11 representation were selected retrospectively on the same cohort; results are exploratory.
