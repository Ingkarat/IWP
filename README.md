# Interprocedural Weakest Precondition

This is a tool that extract hyperparameter constraints from machine-learning operators.

## Installation

The artifact is a Python script which requires Python 3+ environment.

You can install required packages using the package manager [pip](https://pip.pypa.io/en/stable/) and a requirement file.
```bash
pip install -r requirements.txt
```

- scikit-learn is the main target library. The latest version is 1.0.2. The older version 0.24.2 is provided in the artifact too.
- Lale is an IBM's Auto-ML project. We use Lale's hyperopt in the experiment
- black is a code formatter
- jsonschema is an implementation of JSON schema for python
- tabulate is a tabular pretty-printer
- astor is a library for working with Python AST.
- scipy is a computation library. We use scipy to generate sparse datasets.



## Functionality

The tool is a Python script. The main file is **main.py** at the top level directory which should be the only file you need to edit. To run the script, go to this directory and simply run the file with a desired argument that will be explained in following sections. 

```bash
py main.py [argument]
```

The tool has 4 general functionalities.
- Call graph construction
- Weakest precondition analysis
- JSON schema encoder
- Hyperopt experiment

The first 3 functionalities can be used to analyze any Python libraries when providing a class and a function of that class as a target input. The hyperopt experiment tool is for machine-learning libraries, and dataset and some schemas needed to be crafted beforehand. 

Let's look at an example from sklearn's operator **PCA** and a target function **fit**. First you need to provide the appropriate values for the package directory and name at the top of the main function of **main.py**.

```python
def main(argv):
    # package directory and name
    package_dir = "/.../scikit-learn/sklearn"
    package_name = "sklearn"
```

By deafult, using the script without any argument will run an analysis on sklearn PCA's fit function. 

The tool uses colon ":" as a separator throughout the analysis. This means that **package_dir** should not contain ":". For windows, the directory can start with, e.g. **/Users/.../sklearn_0.24/** without the **C:** or **D:** part.

The sklearn directory can be the one you installed (supposedly the latest version), or the older version 0.24.2 provided in the subdirectory *sklearn_0.24*.

### Call graph construction

Call graph construction is a name-based resolution analysis that creates a call graph from a provided target function, e.g. **PCA's fit** function. The command-line argument is **cg**.

```bash
py main.py cg CLASS_NAME FUNCTION_NAME
``` 
Part of the result is shown below. Note that multiple edges from A to B means there are multiple calls from A to B.
```
operator_main_func: /.../decomposition/_pca.py:PCA:fit
operator_main_class: /.../decomposition/_pca.py:PCA
...
Call from  /.../decomposition/_pca.py:PCA:fit  to  /.../decomposition/_pca.py:PCA:_fit
Call from  /.../decomposition/_pca.py:PCA:_fit  to  /.../base.py:BaseEstimator:_validate_data
Call from  /.../decomposition/_pca.py:PCA:_fit  to  /.../decomposition/_pca.py:PCA:_fit_full
...
```

### Weakest precondition analysis
Weakest precondition (WP) analysis is the core analysis that extracts precondition constraints from the code. The analysis also invokes the call graph construction and uses it for the interprocedural analysis. You do not have to run the call graph construction separately before running the WP analysis. 

The text result of the analysis is at */output/(package_name)/text/* directory. It includes WP, soundness flag, a raise exception's information, and a call path from the target function to that exception for each reachable exceptions. Each target function has 2 text result files; True and noTrue.
- [...name...]True =  All WPs are True meaning that their corresponding exceptions will not be raised. This is mostly due to the use of default arguments of functions.
- [...name...]NoTrue = Other WPs that are not True.

Additionally, the WPs constraints are stored in a pickle file (.pkl) in Python AST formula. The location of pickle files is at */output/(package_name)/pkl/*.

The command-line argument is **wp**.

```bash
py main.py wp CLASS_NAME FUNCTION_NAME
``` 
In our example, the pickle file is at */.../output/sklearn/pkl/sklearn_PCA_fit.pkl*

### JSON schema encoder
The encoder encodes precondition constraints from the previous section into JSON schemas. It looks for a pickle file of a target function, encodes, and then formats it using the "black" library. The resulting JSON schemas is at */.../output/(package_name)/jss/*. 

Precondition schemas include both "useful" and "not useful" ones. The "not useful" ones are usually from exceptions nested deep in call chains which could make their precondition constraints long and subjected to path explosion. They are still useful for a quick glance to get an understanding of the exception though. The "useful" ones can be translated to JSON schema and used in standardized tools such as the Lale AutoML library that we use in the hyperopt experiment section.

The command-line argument is **jss**.

```bash
py main.py jss CLASS_NAME FUNCTION_NAME
``` 



### hyperopt experiment (WIP)
The experiment as described in RQ2 of the paper. It generates 1,000 random hyperparameters configurations of an operator and validates them against the precondition constraints that are now in JSON schemas. It is mainly for sklearn and others machine-learning libraries. The handwritten result (from IBM's Lale schemas) that we get here is mostly the same as from our WP constraint because we have contributed PRs to the project to improve their JSON schema constraints. The initial result of handwritten constraints can be found in the paper.

Some operators do not have this experiment due to either the trials exceeded the time limit or they required customized inputs that we could not craft. Some trails still take some time to complete. you can reduce the number of trials by setting **n_trials** variable at the top of **f()** function in **inter_main.py** and **inter_sparse_main.py**.

Note that the error messages here are not actually errors. They are errors from Lale's hyperopt telling us that some configurations of hyperparameters run into runtime errors when calling fit() function.

The command-line argument is **hyperopt**.

```bash
py main.py hyperopt
``` 
For our **PCA** example, we get:
```
400 out of 1000 trials failed, call summary() for details.
...
+-------------------+----------------+-----------------+----------------+-----------------+-----------+--------------------+
|       name        | true_positives | false_positives | true_negatives | false_negatives | precision |       recall       |
+-------------------+----------------+-----------------+----------------+-----------------+-----------+--------------------+
| handwritten_(4/4) |      289       |        0        |      400       |       311       |    1.0    | 0.4816666666666667 |
| docstrings_(0/0)  |      600       |       400       |       0        |        0        |    0.6    |        1.0         |
| WPanalysis_(2/18) |      600       |        0        |      400       |        0        |    1.0    |        1.0         |
+-------------------+----------------+-----------------+----------------+-----------------+-----------+--------------------+
```
This means that for all 400 hyperparameters configurations that run into runtime errors when Lale called PCA's fit() function, these configurations failed to validate against our WPs schemas (at */output/sklearn/jss/sklearn_PCA_fit.py*) and we also said (without running the fit function)that they would fail too. Similarly, the other 600 configurations validate against our WPs schema, which match all the Lale's running trials, resulting in 600 True Positives.

The number (2/18) means that there are 18 WP "not-True" constraints, and 2 of them can be translated into JSON schemas. This can be a little deceiving because a partial schema can fail to validate against a configuration too. In fact, this is the case for one schema shown below. Part of it cannot be translated to JSON so it always validate to TRUE. However, the schema can still be useful and it is responsible for catching 153 configurations that **"n_components='mle' cannot be a string with svd_solver='randomized' or 'arpack'"**. In other words, configurations with hyperparameter "n_components = 'mle'" and "svd_solver = 'randomized' or 'arpack'" will fail to validate against this schema. 

```
        "anyOf": [
            {"type": "object", "properties": {"n_components": {"enum": [None]}}},
            {
                "allOf": [
                    {
                        # This part is long and not expressable, meaning that it always validate to TRUE
                    },
                    {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"svd_solver": {"enum": ["auto"]}},
                            },
                            {
                                "type": "object",
                                "properties": {"svd_solver": {"enum": ["full"]}},
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "svd_solver": {
                                        "not": {"enum": ["arpack", "randomized"]}
                                    }
                                },
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "n_components": {"not": {"type": "string"}}
                                },
                            },
                        ]
                    },
                ]
            },
        ],

```
A detailed summary of each operator trials is in */.../output/sklearn/experiment/result_dense/* directory.

### pipeline option
This is an additional option that run **wp**, **jss** , and **hyperopt** in order. Each input and result is exactly the same as described in previous sections.

The command-line argument is **pipeline**.

```bash
py main.py pipeline
``` 

## RQ1: Does the analysis find real issues?

We look at one specific exception.
```python
def _ensure_sparse_format(...):
    ...
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
```
As explained in the paper, this exception is guarded by many conditionals along call chains starting at **fit** or **predict** function. We are interested in constraints in this form.
```
(hasattr(X,'sparse') and X.ndim > 1  =>  NOT(sp.issparse(X.sparse.to_coo()))) and (NOT(hasattr(X,'sparse') and X.ndim > 1)  =>  NOT(sp.issparse(X)))
```
At the top of **fit** or **predict** function of an operator:

1. If the constraint is in the above format, then this target function does not support sparse input because this exception will be raised.
2. If the constraint is **TRUE**, then this target function accepts sparse input.

We then compare with the documentation and discovered 2 inconsistency of case (1) and 22 instances of case (2).

To run the analysis, the command-line argument is **rq1**.
```bash
py main.py rq1
```
This will run the analysis on sklearn version 0.24.2 on this specific exception.

The summary file in */.../output/rq1/summary* shows the number of total, True, and non-True constraints. Multiple constraints means there are multiple call paths from the target function to this raise statement. The subdirectory */result/* stores detailed information of each operator in a similar format in the regular WP analysis.

For case (1), there are **AffinityPropagation's predict** (*\...\output\rq1\result\[sklearn_AffinityPropagation_predict]NoTrue*) and **MeanShift's predict** (*\...\output\rq1\result\[sklearn_MeanShift_predict]NoTrue*) function. 

For **AffinityPropagation**, a [PR](https://github.com/scikit-learn/scikit-learn/pull/20117) was created with a fix to accept sparse input.

For **MeanShift**, this led to a discussion among sklearn developers and eventually settled on a documentation fix ([PR](https://github.com/scikit-learn/scikit-learn/pull/20796)). The documentation [Before](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation.predict) and [After](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation.predict).



## RQ3: Does Soundness Analysis Work?
We look at soundness of preconditions constraints at the top of operators' fit functions. There is **one single For loop** in the shared data validation code that is responsible for most of the unsoundness. Weakest preconditions for For loop is not computable in general and the analysis reasons about what variables are modified with in a loop to limit the impact of the loop. The paper discusses this specific loop in details. By modifying them to the **If statement** that has the same functionality, we achieve high soundness of the inferred preconditions.

To run the analysis, the command-line argument is **rq3**.
```bash
py main.py rq3
```
This will run the WP analysis on the operators' fit function from the  0.24.2 sklearn (in */.../sklearn/*), as well as on the modified version (in */.../etc/sklearn_0.24_modified*) that changed the loop into the If statement. The summary results are in */.../output/rq3/rq3_original_stats* and in */.../output/rq3/rq3_modified_stats* respectively.
They are in the format **[ARDRegression,104,82,22,13,9]** meaning that **ARDRegression's fit** function has 104 preconditions; 82 are **True** and 22 are **non-True**. For the non-True preconditions, 13 are sound and 9 are unsound.

Because this script executes a full weakest precondition analysis twice for each operator, it takes some time for a complete run. Below is an example of the summary file.
* Original sklearn
```
[ARDRegression,104,82,22,13,9]
[AdaBoostClassifier,134,90,44,35,9]
[AdaBoostRegressor,155,98,57,48,9]
[AdditiveChi2Sampler,144,139,5,3,2]
[AffinityPropagation,200,178,22,11,11]
[AgglomerativeClustering,155,141,14,9,5]
[BaggingClassifier,140,69,71,36,35]
[BaggingRegressor,163,105,58,30,28]
[BayesianRidge,117,84,33,21,12]
...
```
* Modified sklearn
```
[ARDRegression,122,94,28,28,0]
[AdaBoostClassifier,157,105,52,52,0]
[AdaBoostRegressor,178,113,65,65,0]
[AdditiveChi2Sampler,150,140,10,10,0]
[AffinityPropagation,227,188,39,39,0]
[AgglomerativeClustering,167,145,22,22,0]
[BaggingClassifier,169,88,81,57,24]
[BaggingRegressor,202,134,68,51,17]
[BayesianRidge,140,100,40,40,0]
...
```


Note that this script only makes the summary files. It will not create pickle files or text files from the WP analysis. 

## RQ 4
To run the analysis, the command-line argument is **rq3**.
```bash
py main.py rq3
```
We run the analysis on selected operators to show the impact of the Concrete Evaluation (CE). Each operator is run twice with and without CE. We also remove the pruning heuristic that removes constraints that are very large, specifically with 200 implications or more.

The result is in */.../rq4/pkl/* for pickle files. The running time is shown on the command prompt.

## RQ2
The dense dataset can be run using the **pipeline** argument or an individual **hyperopt** argument if schema is created separately. The instruction and location of the results are the same as in the *hyperopt experiment* section.

Note that some results are different from the paper because Lale's run against the latest (sklearn) fit() function while our JSON schema from the analysis is from the older version. There is also a slight change in the analysis too

## Note
1. There is another variable **skip_run_if_required_file_exists** at the top of **main.py** that can be configured. The default is False. If set to True, the tool will try to skip a run if possible. This does not work on all running options.
2. While running the tool, you may encounter the warning message *<string>:1: SyntaxWarning: 'list' object is not callable; perhaps you missed a comma?*. This is from the **exec()** function where the analysis tries to interpret the node natively. 
