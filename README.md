# Machine Learning Projects — Solved Notebooks

Five completed machine learning coursework notebooks with implementations, mathematical explanations, assertions, and reproducible experiments. The original exercise order is retained; code cells and written-answer cells contain solutions.

| Notebook | Dataset | Implemented methods |
|---|---|---|
| [Linear regression](notebooks/Linear_Regression.ipynb) | Combined Cycle Power Plant | SVD pseudoinverse, least squares, RSS and R², prediction and coefficient intervals, feature selection |
| [Linear classification](notebooks/Linear_Classification.ipynb) | Wine | Constrained splits, pocket perceptron, binary and multiclass logistic regression, decision regions |
| [Regularization](notebooks/Regression_Regularization.ipynb) | King County housing | Best-subset selection, closed-form Ridge, fold-local cross-validation, Lasso paths, redundant-feature experiments |
| [SVMs and neural networks](notebooks/SVM_and_NNs.ipynb) | MNIST | Five-fold kernel/architecture search, confusion matrices, learning curves, large-data classification, matched scaling experiment |
| [Unsupervised learning](notebooks/Unsupervised_Learning.ipynb) | 1-D data, synthetic blobs, MNIST | Gaussian density, E/M updates, stable mixture likelihood, GMM and K-means, training-only cluster-label mapping |

## Run

Python 3.12 was used for verification. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`  
macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook, restart its kernel, and run all cells in order. Some cells intentionally add an intercept or rescale an existing variable, so selectively rerunning those cells is not equivalent to a fresh execution.

The project uses `ID_number = 42` as a random seed, **not an invented university registration number**. For a course submission, replace it with your real matricola and use the course's required filenames; changing the seed changes the results.

## Data and computational cost

The original dataset-loading URLs remain in the notebooks. CSV data are read from the referenced course repository; Wine is bundled with scikit-learn; MNIST is downloaded through OpenML and cached by scikit-learn. Internet access is needed for the initial remote downloads. Datasets are not included in this repository.

The complete MNIST dataset contains 70,000 images. The SVM notebook keeps the requested 500-example model search and the 60,000-example large training split. The clustering notebook keeps 5,000 training images, 65,000 test images, and full-covariance Gaussian mixtures. These are full experiments, not small substitutes; allow several minutes or longer depending on your CPU and memory.

## Verification

See [execution results](docs/RESULTS.md) for measured outcomes and [implementation notes](docs/IMPLEMENTATION_NOTES.md) for corrections to the templates.

Run the focused offline numerical checks:

```bash
python -m unittest discover -s tests -v
```

A script is included for sequential execution without a Jupyter server; it embeds figures and outputs back into the selected notebook:

```bash
python scripts/execute_notebook.py notebooks/Linear_Regression.ipynb
```

## Publish to GitHub

Create an empty repository named `machine-learning-projects`, then run from this directory:

```bash
git init -b main
git add .
git commit -m "Add solved machine learning notebooks and verified experiments"
git remote add origin https://github.com/YOUR_USERNAME/machine-learning-projects.git
git push -u origin main
```

Replace `YOUR_USERNAME`. If the repository already has content, clone it and copy these files into it instead of initializing another repository.

## Attribution

The supplied notebooks include course-authored instructions and scaffolding. Their original data references are retained. Solutions and explanatory additions were prepared with AI assistance. No redistribution license or ownership is asserted over the original course material or datasets.
