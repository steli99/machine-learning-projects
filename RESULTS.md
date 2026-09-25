# Verified execution results

All five completed notebooks ran from their first code cell to their last code cell on the original datasets, with seed 42. Outputs and plots are embedded in each notebook. No reduced-data substitute or skipped model-training cell was used.

| Notebook | Result | Elapsed seconds |
|---|---|---:|
| Unsupervised_Learning | Passed | 239.0 |
| Regression_Regularization | Passed | 68.6 |
| Linear_Regression | Passed | 11.8 |
| SVM_and_NNs | Passed | 457.6 |
| Linear_Classification | Passed | 18.9 |

## Selected measured results

| Experiment | Held-out metric |
|---|---:|
| Power plant, full least squares | R² 0.9312 |
| Power plant, validation-selected subset | R² 0.9307 |
| Wine, binary logistic regression | Accuracy 93.88% |
| Wine, three-class logistic regression | Accuracy 93.88% |
| Housing, full least squares | R² 0.5118 |
| Housing, manual four-feature model | R² 0.6082 |
| Housing, best subset | R² 0.4871 |
| Housing, CV Ridge | R² 0.4699 |
| Housing, CV Lasso | R² 0.5702 |
| MNIST, SVM trained on 500 | Accuracy 89.40% |
| MNIST, selected MLP trained on 500 | Accuracy 85.43% |
| MNIST, ReLU MLP trained on 60,000 | Accuracy 96.38% |
| MNIST, GMM trained on 5,000 | Accuracy 42.87% |
| MNIST, K-means trained on 5,000 | Accuracy 58.44% |
| MNIST, MLP trained on 5,000 | Accuracy 93.79% |

The MNIST test partitions and training sizes differ across experiments; these rows are not all equal-data comparisons. Housing has only 100 training observations, so its ranking is sensitive to the seed and sample. Written answers distinguish empirical findings from guarantees.

All nine offline test cases passed. They cover rank-deficient pseudoinverses, disjoint splitting, prediction metrics, unpenalized Ridge intercepts, perceptron convergence and pocket behavior, confusion matrices with predicted-only classes, Gaussian density and stable responsibilities, EM likelihood/final posterior consistency, and training-only cluster-label mapping. See `tests.txt`.

## Runtime environment

Exact tested versions are in `environment.json` and the numerical dependencies are pinned in `requirements.txt`. Individual execution JSON files preserve measured results and timing.
