Good reference

https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc

https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search/53292354

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
