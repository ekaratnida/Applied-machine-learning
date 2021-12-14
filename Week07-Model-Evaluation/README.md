<H1> Good reference </H1>

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


<H2> Jaccard-based FBeta score </H2>

<pre>
The objective of the competition is to identify the mention of datasets within scientific publications. Your predictions will be short excerpts from the publications that appear to note a dataset.

Submissions are evaluated on a Jaccard-based FBeta score between predicted texts and ground truth texts, with Beta = 0.5 (a micro F0.5 score). Multiple predictions are delineated with a pipe (|) character in the submission file.

The following is Python code for calculating the Jaccard score for a single prediction string against a single ground truth string. Note that the overall score for a sample uses Jaccard to compare multiple ground truth and prediction strings that are pipe-delimited - this code does not handle that process or the final micro F-beta calculation.

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
Note that ALL ground truth texts have been cleaned for matching purposes using the following code:

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
For each publication's set of predictions, a token-based Jaccard score is calculated for each potential prediction / ground truth pair. The prediction with the highest score for a given ground truth is matched with that ground truth.

Predicted strings for each publication are sorted alphabetically and processed in that order. Any scoring ties are resolved on the basis of that sort.
Any matched predictions where the Jaccard score meets or exceeds the threshold of 0.5 are counted as true positives (TP), the remainder as false positives (FP).
Any unmatched predictions are counted as false positives (FP).
Any ground truths with no nearest predictions are counted as false negatives (FN).
All TP, FP and FN across all samples are used to calculate a final micro F0.5 score. (Note that a micro F score does precisely this, creating one pool of TP, FP and FN that is used to calculate a score for the entire set of predictions.)
</pre>

Evidently Ai
https://m.youtube.com/watch?v=L4Pv6ExBQPM&feature=youtu.be

## Good example of precision and recall

"For rare cancer data modeling, anything that doesn't account for false-negatives is a crime. Recall is a better measure than precision.
For YouTube recommendations, false-negatives is less of a concern. Precision is better here." https://datascience.stackexchange.com/a/30882

"We have thousands of free customers registering in our website every week. The call center team wants to call them all, but it is imposible, so they ask me to select those with good chances to be a buyer (with high temperature is how we refer to them). We don't care to call a guy that is not going to buy (so precision is not important) but for us is very important that all of them with high temperature are always in my selection, so they don't go without buying. That means that my model needs to have a high recall, no matter if the precision goes to hell." https://datascience.stackexchange.com/a/30884

https://forecasters.org/wp-content/uploads/gravity_forms/7-621289a708af3e7af65a7cd487aee6eb/2015/07/Kolassa_Stephan_ISF2015.pdf

https://blog.clairvoyantsoft.com/churning-the-confusion-out-of-the-confusion-matrix-b74fb806e66

Recall: https://c3.ai/glossary/data-science/recall/
