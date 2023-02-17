<H1> Good reference </H1>

Feature selection </br>
- https://arxiv.org/pdf/2005.12483.pdf
- https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc
- https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search/53292354
- https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
- https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
- https://towardsdatascience.com/boruta-shap-an-amazing-tool-for-feature-selection-every-data-scientist-should-know-33a5f01285c0
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel

<pre>
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
iris_data = load_iris()
X, y = iris_data.data, iris_data.target


# Just initialize the pipeline with any estimator you like    
pipe = Pipeline(steps=[('estimator', SVC())])

# Add a dict of estimator and estimator related parameters in this list
params_grid = [{
                'estimator':[SVC()],
                'estimator__C': [1, 10, 100, 1000],
                'estimator__gamma': [0.001, 0.0001],
                },
                {
                'estimator': [DecisionTreeClassifier()],
                'estimator__max_depth': [1,2,3,4,5],
                'estimator__max_features': [None, "auto", "sqrt", "log2"],
                },
               # {'estimator':[Any_other_estimator_you_want],
               #  'estimator__valid_param_of_your_estimator':[valid_values]

              ]

grid = GridSearchCV(pipe, params_grid)
</pre>

https://www.bualabs.com/archives/532/what-is-training-set-why-train-test-split-training-set-validation-set-test-set/

Evidently Ai
https://m.youtube.com/watch?v=L4Pv6ExBQPM&feature=youtu.be

## Good example of precision and recall
- Example
  - "For rare cancer data modeling, anything that doesn't account for false-negatives is a crime. Recall is a better measure than precision.
  - For YouTube recommendations, false-negatives is less of a concern. Precision is better here." https://datascience.stackexchange.com/a/30882
- https://forecasters.org/wp-content/uploads/gravity_forms/7-621289a708af3e7af65a7cd487aee6eb/2015/07/Kolassa_Stephan_ISF2015.pdf
- https://blog.clairvoyantsoft.com/churning-the-confusion-out-of-the-confusion-matrix-b74fb806e66
- Recall: https://c3.ai/glossary/data-science/recall/
- K-Folds: https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/
- https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate.
- https://en.wikipedia.org/wiki/Precision_and_recall
- https://stackoverflow.com/questions/26355942/why-is-the-f-measure-a-harmonic-mean-and-not-an-arithmetic-mean-of-the-precision
- https://en.wikipedia.org/wiki/Harmonic_mean

## Multi-class
- https://vitalflux.com/micro-average-macro-average-scoring-metrics-multi-class-classification-python/
- https://datascience.stackexchange.com/a/24051

## ROC AUC
- https://towardsdatascience.com/understanding-the-roc-curve-in-three-visual-steps-795b1399481c
- https://github.com/akshaykapoor347/Compute-AUC-ROC-from-scratch-python/blob/master/AUCROCPython.ipynb
- https://paulvanderlaken.com/2019/08/16/roc-auc-precision-and-recall-visually-explained/
- https://datascience.stackexchange.com/a/24051
- https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/

## SHAP
- https://www.kaggle.com/learn/machine-learning-explainability
