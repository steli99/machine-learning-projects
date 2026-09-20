"""Focused numerical checks of the notebook implementations, without downloading data."""
import ast
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

NOTEBOOKS = Path(__file__).resolve().parents[1] / 'notebooks'

def functions_from(name):
    namespace = {'np': np, 'pd':pd, 'ID_number': 42, 'logsumexp': logsumexp,
                 'linear_model': linear_model}
    notebook = json.loads((NOTEBOOKS / f'{name}.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=funcs, type_ignores=[]), name, 'exec'), namespace)
    return namespace

class AlgorithmTests(unittest.TestCase):
    def test_rank_deficient_pseudoinverse(self):
        f = functions_from('Linear_Regression')
        A = np.array([[1.,2.,3.],[2.,4.,6.]])
        inverse = f['pseudoinverse'](A)
        np.testing.assert_allclose(inverse, np.linalg.pinv(A), atol=1e-12)
        np.testing.assert_allclose(A @ inverse @ A, A)
        np.testing.assert_allclose(f['pseudoinverse'](np.zeros((3,2))), np.zeros((2,3)))

    def test_split_is_disjoint_and_complete(self):
        f = functions_from('Linear_Regression')
        data = np.arange(60).reshape(20,3)
        splits = f['create_train_val_test_datasets'](data,7,6,7)
        self.assertEqual(len(np.unique(np.vstack(splits)[:,0])),20)
        np.testing.assert_array_equal(data,np.arange(60).reshape(20,3))

    def test_prediction_and_metrics(self):
        f = functions_from('Linear_Regression')
        x = np.arange(4.).reshape(-1,1); w = np.array([[2.],[3.]])
        y = 2+3*x
        np.testing.assert_allclose(f['linear_predictions'](w,x),y)
        self.assertEqual(f['cod'](y,y),1.)
        self.assertEqual(f['rss'](y,y.ravel()),0.)

    def test_ridge_matches_sklearn_with_noncentered_data(self):
        f = functions_from('Regression_Regularization')
        rng = np.random.default_rng(8); x = rng.normal(3,2,(50,4)); y = 7+x @ np.arange(4)+rng.normal(size=50)
        for alpha in [0.,0.1,100.]:
            w = f['compute_LS_optimal_ridge_ERM_coefficients'](x,y,alpha)
            model = linear_model.Ridge(alpha=alpha).fit(x,y)
            np.testing.assert_allclose(w,np.r_[model.intercept_,model.coef_],rtol=1e-9,atol=1e-9)

    def test_perceptron_separable_and_pocket(self):
        f = functions_from('Linear_Classification'); np.random.seed(42)
        x = np.array([[1,-2],[1,-1],[1,1],[1,2]],float); y = np.array([-1,-1,1,1])
        w,error,wb,eb,ws,es = f['perceptron'](x,y,100)
        self.assertEqual(error,0);np.testing.assert_array_equal(np.sign(x @ w),y)
        self.assertTrue(np.all(np.diff(eb)<=0))
        self.assertEqual(len(wb),len(ws));self.assertEqual(len(es),len(eb))
        self.assertEqual(f['choose_missclassified'](np.array([])),-1)
        # Contradictory labels cannot be separated; verify termination and pocket scoring.
        result = f['perceptron'](np.ones((2,2)),np.array([-1,1]),15)
        self.assertEqual(len(result[4]),15)
        self.assertEqual(result[1],f['compute_fraction_missclassified'](np.ones((2,2)),np.array([-1,1]),result[0]))

    def test_confusion_matrix_includes_predicted_only_class(self):
        f = functions_from('SVM_and_NNs')
        y,p = np.array(['a','b','a']),np.array(['c','b','a'])
        np.testing.assert_array_equal(f['confusion_matrix_by_hand'](y,p),confusion_matrix(y,p))

    def test_gaussian_density_and_responsibilities(self):
        f = functions_from('Unsupervised_Learning')
        x = np.array([1.,2.]); mu=np.array([0.,1.]); cov=np.array([[2.,0.3],[0.3,1.]])
        self.assertAlmostEqual(f['gaussian_pdf'](x,mu,cov).item(),multivariate_normal.pdf(x,mu,cov))
        means=np.array([[0.,0.],[3.,3.]]); covs=np.array([np.eye(2),np.eye(2)])
        W=f['E_step'](np.array([[1000.,1000.],[0.,0.]]),np.array([.5,.5]),means,covs)
        self.assertTrue(np.isfinite(W).all());np.testing.assert_allclose(W.sum(1),1.)

    def test_em_monotonicity_and_final_posterior(self):
        f = functions_from('Unsupervised_Learning');np.random.seed(42)
        rng=np.random.default_rng(10)
        data=np.r_[rng.normal(-3,.5,(60,1)),rng.normal(3,.5,(60,1))]
        W,pi,means,covs,history=f['run_EM_on_GMM'](data,2,100)
        self.assertTrue(np.all(np.diff(history)>=-1e-6))
        np.testing.assert_allclose(W,f['E_step'](data,pi,means,covs))
        np.testing.assert_allclose(pi.sum(),1.)

    def test_cluster_mapping_fixed_for_test(self):
        f=functions_from('Unsupervised_Learning')
        mapping=f['cluster_label_mapping'](np.array(['a','a','b']),np.array([1,1,0]),3)
        np.testing.assert_array_equal(mapping,np.array(['b','a','a']))
        # Reversing test labels must count as errors, not redefine the mapping.
        self.assertEqual(f['compute_score'](np.array(['a','b']),np.array([0,1]),3,mapping),1.)

if __name__ == '__main__':
    unittest.main()
