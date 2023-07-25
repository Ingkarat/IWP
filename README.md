# Interprocedural Weakest Precondition

This is a tool that extracts hyperparameter constraints from machine-learning operators.

## Installation

The artifact is a Python script which requires a Python 3+ environment.

You can install required packages using the package manager [pip](https://pip.pypa.io/en/stable/) and a requirement file.
```bash
pip install -r requirements.txt
```

- scikit-learn is the main target (and default) library. The latest version is 1.0.2. (The older version 0.24.2 was used in the experiment)
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
- Hyperopt experiment (WIP)

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

The text result of the analysis is at */output/(package_name)/text/* directory. It includes WP, soundness flag, a raise exception's information, and a call path from the target function to that exception for each reachable exception. Each target function has 2 text result files; True and noTrue.
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

Precondition schemas include both "useful" and "not useful" ones. The "not useful" ones are usually from exceptions nested deep in call chains which could make their precondition constraints long and subject to path explosion. They are still useful for a quick glance to get an understanding of the exception though. The "useful" ones can be translated to JSON schema and used in standardized tools such as the Lale AutoML library that we use in the hyperopt experiment section.

The command-line argument is **jss**.

```bash
py main.py jss CLASS_NAME FUNCTION_NAME
``` 

#### Configuration
To customize the tool, additional configurations and their descriptions are available in **main.py** under *"# MISC setting"*.


## Impact
The IWP is effective at finding real issues.

- sklearn
    - Operators support sparse inputs but the code throws exceptions
        - [AffinityPropagation](https://github.com/scikit-learn/scikit-learn/issues/20049)
        - [MeanShift](https://github.com/scikit-learn/scikit-learn/pull/20117)
    - Undocumented hyperparameter constraints
        - [Multiple Operators](https://github.com/scikit-learn/scikit-learn/pull/19444)
- imblearn
    - Type discrepancy between documentation and precondition
        - [InstanceHardnessThreshold](https://github.com/scikit-learn-contrib/imbalanced-learn/issues/889)
- TensorFlow
    - Discrepancy between documentation and preconditions
        - [nn.gelu](https://github.com/tensorflow/tensorflow/issues/57965)
        - [nn.ctc_loss](https://github.com/tensorflow/tensorflow/issues/57964)
    - Undocumented explicit padding exceptions
        - [nn.max_pool and nn.max_pool2d](https://github.com/tensorflow/tensorflow/issues/57978) 
- NumPy
    - Missing exception in documentation
        - [memmap](https://github.com/numpy/numpy/issues/22643)

- [Lale](https://github.com/IBM/lale): an IBM's Auto-ML project 
    - Add missing schema constraints
        - [DummyRegressor](https://github.com/IBM/lale/pull/767)
        - [LightGBM](https://github.com/IBM/lale/pull/762)
    - Improve schema for Hand-Written and NL Docstrings constraints
        - [BernoulliNB](https://github.com/IBM/lale/pull/818)
        - [NL Docstrings 1](https://github.com/IBM/lale/pull/821), [NL Docstrings 2](https://github.com/IBM/lale/pull/826), [Hand-Written](https://github.com/IBM/lale/pull/832)


### Libraries
The IWP is general and can run on any library when provided with an (operator class, target method) entry tuple. 
We started with **scikit-learn** as the target library, and then applied the analysis to others 10 libraries.

The 11 libraries can be categorized into two groups:
- sklearn-compatible libraries: sklearn, XGBoost, LightGBM, imblearn, category_encoders, MAPIE, metric_learn, and sklearn_pandas
- other libraries: TensorFlow, NumPy, and Pandas