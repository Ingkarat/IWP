import sklearn.datasets
import sklearn.model_selection
import collections
import pandas as pd
import pprint
import lale.operators
import ast
import jsonschema
import sys
import pathlib
import json
import traceback
import warnings
import logging
import scipy
from tabulate import tabulate

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

import getlale
import checkConstNum

from lale.lib.lale import Hyperopt, NoOp, ConcatFeatures
from lale.lib.sklearn import LogisticRegression as LR_handwritten
from lale.lib.sklearn import MissingIndicator, OrdinalEncoder, LogisticRegression
from lale.lib.sklearn import MinMaxScaler, NMF
from lale.lib.sklearn import VotingClassifier, KNeighborsClassifier, DecisionTreeClassifier
from lale.lib.sklearn import ColumnTransformer, PCA
from lale.lib.sklearn import Pipeline
from lale.lib.sklearn import RFE
from lale.lib.sklearn import BaggingClassifier
from lale.lib.sklearn import FunctionTransformer
from lale.lib.sklearn import StackingClassifier, StackingRegressor, SGDRegressor

Data = collections.namedtuple("Data", ["train_X", "test_X", "train_y", "test_y"])

# ----- load Iris dataset
# - do a train/test split in case we want to evaluate predictive performance
# - put this into a named tuple in case we want to play with different datasets
def fetch_iris():
    all_X, all_y = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
        all_X, all_y, test_size=0.3, random_state=42)
    result = Data(train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y)
    return result

def fetch_regression_data():
	all_X, all_y = sklearn.datasets.make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
	train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
        all_X, all_y, test_size=0.3, random_state=42)
	result = Data(train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y)
	return result

def fetch_multi():
	X_multi = [[i, i] for i in range(100)]
	result = Data(train_X=X_multi, test_X=[], train_y=X_multi, test_y=[])
	return result

LR = LR_handwritten.customize_schema(relevantToOptimizer=['solver', 'penalty', 'dual', 'C', 'tol', 'fit_intercept',
        'intercept_scaling', 'max_iter', 'multi_class', 'warm_start', 'l1_ratio'])

iris = fetch_iris()
regression_data = fetch_regression_data()
multi_data = fetch_multi()

def f_write_schema_to_file(HW, Uncntr, Docstr, WP):
	with open("tt1HW.py", "w") as f:
		if HW is not None:
			print(pprint.pformat(HW._schemas, width = 200, sort_dicts = False), file = f)
	with open("tt2Uncntr.py", "w") as f:
		print(pprint.pformat(Uncntr._schemas, width = 200, sort_dicts = False), file = f)
	with open("tt3Docstr.py", "w") as f:
		if Docstr is None:
			print("no lale autogen", file = f)
		else:
			print(pprint.pformat(Docstr._schemas, width = 200, sort_dicts = False), file = f)
	with open("tt4WP.py", "w") as f:
		print(pprint.pformat(WP._schemas, width = 200, sort_dicts = False), file = f)

# ----- inspect examples of one successful and one failed run
def f_sanity_check(rand_trained, x):
	print("\n === BEGIN sanity check ===")
	print("op = ", x)
	example_success = None
	example_failure = None
	for name in rand_trained.summary().index:
	    status = rand_trained.summary().at[name, "status"]
	    if status == "ok":
	        if example_success is None:
	            example_success = rand_trained.get_pipeline(name)
	    else:
	        if example_failure is None:
	            example_failure = rand_trained.get_pipeline(name)

	print(f"example success: {example_success.hyperparams()}")
	# sanity check: the following should not raise an exception
	_ = example_success.fit(iris.test_X, iris.test_y)

	print(f"example failure: {example_failure.hyperparams()}")
	try:
	    _ = example_failure.fit(iris.test_X, iris.test_y)
	except ValueError as e:
	    expected = "Solver newton-cg supports only dual=False, got dual=True"
	    assert str(e) == expected, f"got {e.message}, expected {expected}"	
	print(" === END sanity check ===")

def replace_constraints(op_orig, constraints):
    schema_orig = op_orig._schemas
    base_hyperparams = schema_orig['properties']['hyperparams']['allOf'][0]
    schema_result = {
        **schema_orig,
        'properties': {
            **schema_orig['properties'],
            'hyperparams': {
                'allOf': [
                    base_hyperparams,
                    *constraints]}}}
    op_result = lale.operators.make_operator(op_orig._impl_class(), schema_result)
    return op_result

def get_constraints(op):
    result = op.hyperparam_schema()["allOf"][1:]
    return result

def gg(a,b):
	print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
	v = jsonschema.Draft4Validator(b)
	errors = sorted(v.iter_errors(a), key=lambda e: e.path)
	for error in errors:
		for suberror in sorted(error.context, key=lambda e: e.schema_path):
			print(list(suberror.schema_path), suberror, sep=",")

	print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
	jsonschema.validate(a, b, jsonschema.Draft4Validator)

def ensure_dense(X):
    if isinstance(X, scipy.sparse.csr_matrix):
        return X.toarray()
    return X

all_hp = []
failed_schema_hw = []
failed_schema_hw_dict = {}
failed_hp_trials_hw_dict = {}
failed_schema_auto = []
failed_schema_auto_dict = {}
failed_hp_trials_auto_dict = {}
failed_schema_wp = []
failed_schema_wp_dict = {}
failed_hp_trials_wp_dict = {}

print_hw_const = {}
print_auto_const = {}
print_wp_const = {}
all_trials_hp = []
hw_failed_trials_hp = []
auto_failed_trials_hp = []
wp_failed_trials_hp = []

hw_TruePos = []
hw_FalsePos = []
hw_TrueNeg = []
hw_FalseNeg = []
auto_TruePos = []
auto_FalsePos = []
auto_TrueNeg = []
auto_FalseNeg = []
wp_TruePos = []
wp_FalsePos = []
wp_TrueNeg = []
wp_FalseNeg = []

hp_distinction = {}

global_f_train_X = None
global_f_train_y = None


def status_for_schema(config, op, printthis, c_type, schema_type, pipelinePos):
	if op is None:
		#print("NNNNNNNNNNNNNNNNNN")
		return "None"

	if pipelinePos == 1:
		hyperparams = config.steps()[1]._get_params_all()
		schema = op.hyperparam_schema()
	elif pipelinePos == 90: #VotingClassifier
		hyperparams = config._get_params_all()
		schema = op.hyperparam_schema()
		hyperparams["estimators"] = list(hyperparams["estimators"])
		for ii in range(len(hyperparams["estimators"])):
			hyperparams["estimators"][ii] = list(hyperparams["estimators"][ii])	
	elif pipelinePos == 91: #ColumnTransformer
		hyperparams = config.steps()[0]._get_params_all()
		schema = op.hyperparam_schema()
		for ii in range(len(hyperparams["transformers"])):
			hyperparams["transformers"][ii] = list(hyperparams["transformers"][ii])
	elif pipelinePos == 92: #Pipeline
		hyperparams = config._get_params_all()
		schema = op.hyperparam_schema()
		for ii in range(len(hyperparams["steps"])):
			hyperparams["steps"][ii] = list(hyperparams["steps"][ii])
	elif pipelinePos == 93: #StackingClassifier
		hyperparams = config._get_params_all()
		schema = op.hyperparam_schema()	
	elif c_type == "transformer":
		#hyperparams = config.steps()[0].hyperparams()
		hyperparams = config.steps()[0]._get_params_all()
		schema = op.hyperparam_schema()
	else:
		hyperparams = config._get_params_all()
		schema = op.hyperparam_schema()

	if "quantile_range" in hyperparams:
		hyperparams["quantile_range"] = list(hyperparams["quantile_range"])

	# Check hyperparams (hand-inspect)
	#print(schema_type, "hyperparams =",hyperparams)
	#print("schema =", schema)
	if 1:
		for h in hyperparams:
			if h not in hp_distinction:
				hp_distinction[h] = [hyperparams[h]]
			else:
				if hyperparams[h] not in hp_distinction[h]:
					hp_distinction[h].append(hyperparams[h])
		if 0: #RFE
			if hyperparams["n_features_to_select"] is not None or hyperparams["step"] != 1 or hyperparams["verbose"] != 0:
				print(hyperparams)
				sys.exit("AAAAAAA")
		if 0: #Pipeline
			if hyperparams["memory"] is not None or hyperparams["verbose"]:
				sys.exit("AAAAAAA")
		if 0: #ColumnTransformer
			if hyperparams["remainder"] != "drop" or hyperparams["sparse_threshold"] != 0.3 or hyperparams["n_jobs"] is not None or hyperparams["transformer_weights"] is not None or hyperparams["verbose"]:
				print(hyperparams)
				sys.exit("AAAAAAA")
		if 0: #VotingClassifier
			if not hyperparams["flatten_transform"]:
				sys.exit("AAAAAAA")
		if 0: #NMF
			if hyperparams["solver"] != "cd" or hyperparams["beta_loss"] != "frobenius":
				sys.exit("AAAAAAA")

	if "hidden_layer_sizes" in hyperparams:
		hyperparams["hidden_layer_sizes"] = list(hyperparams["hidden_layer_sizes"])
	if "feature_range" in hyperparams:
		hyperparams["feature_range"] = list(hyperparams["feature_range"])

		
	# Get all hp when reaching this for the first time
	if not all_hp:
		for k in hyperparams:
			if k not in all_hp:
				all_hp.append(k)
				failed_hp_trials_hw_dict[k] = 0
				failed_hp_trials_auto_dict[k] = 0
				failed_hp_trials_wp_dict[k] = 0

	if printthis:
		with open("tt998hp.py", "a+") as f:
			print(pprint.pformat(hyperparams, width = 200, sort_dicts = False), file = f)
		if 0:
			with open("tt999schema.py", "a+") as f:
				print(pprint.pformat(schema, width = 200, sort_dicts = False), file = f)

		#gg(hyperparams, schema)

	if 0:
		try:
			validator = jsonschema.validate(hyperparams, schema, jsonschema.Draft4Validator)
			result = "ok"
		except jsonschema.ValidationError:
			result = "fail"

	const_sch = ""
	try:
		hp_all = hyperparams
		hp_schema = schema
		data_schema = lale.helpers.fold_schema(global_f_train_X, global_f_train_y)
		hp_schema_2 = lale.type_checking.replace_data_constraints(
			hp_schema, data_schema
		)
		const_sch = ""
		# def validate_schema(value, schema: JSON_TYPE, subsample_array: bool = True):
		lale.type_checking.validate_schema(hyperparams, hp_schema_2, False)
		result = "ok"
	except jsonschema.ValidationError as e_orig:
		e = e_orig if e_orig.parent is None else e_orig.parent
		const_sch = e.schema
		result = "fail"
		if 0:
			if schema_type == "wp":
				print(const_sch)
				print("=========")
	except:
		if 0:
			e = sys.exc_info()[1]
			str_e = str(e)
			exception_type = f"{type(e).__module__}.{type(e).__name__}"
			error_msg = f"Exception caught: {exception_type}, {traceback.format_exc()}"
			if schema_type == "wp":
				print("555", error_msg)
		const_sch = "Error that is not ValidationError"
		result = "fail"

	if result == "fail":
		if schema_type == "hw":
			if const_sch not in failed_schema_hw:
				failed_schema_hw.append(const_sch)
			idx = str(failed_schema_hw.index(const_sch))
			if idx in failed_schema_hw_dict:
				failed_schema_hw_dict[idx] += 1
			else:
				failed_schema_hw_dict[idx] = 1
		elif schema_type == "auto":
			if const_sch not in failed_schema_auto:
				failed_schema_auto.append(const_sch)
			idx = str(failed_schema_auto.index(const_sch))
			if idx in failed_schema_auto_dict:
				failed_schema_auto_dict[idx] += 1
			else:
				failed_schema_auto_dict[idx] = 1
		elif schema_type == "wp":
			if const_sch not in failed_schema_wp:
				failed_schema_wp.append(const_sch)
			idx = str(failed_schema_wp.index(const_sch))
			if idx in failed_schema_wp_dict:
				failed_schema_wp_dict[idx] += 1
			else:
				failed_schema_wp_dict[idx] = 1

	if schema_type == "hw":
		global print_hw_const
		if not print_hw_const:
			print_hw_const = schema
		#all_trials_hp.append(hyperparams)
		if result == "fail":
			hw_failed_trials_hp.append(hyperparams)

	elif schema_type == "auto":
		global print_auto_const
		if not print_auto_const:
			print_auto_const = schema
		#all_trials_hp.append(hyperparams)
		if result == "fail":
			auto_failed_trials_hp.append(hyperparams)

	elif schema_type == "wp":
		global print_wp_const
		if not print_wp_const:
			print_wp_const = schema
		all_trials_hp.append(hyperparams)
		if result == "fail":
			wp_failed_trials_hp.append(hyperparams)

	# Get validation stats
	#print(">>>>>>>>>>>")
	if 0:
		validator = jsonschema.Draft4Validator(schema)
		errors = list(validator.iter_errors(hyperparams))

		if (not errors) and (result == "fail"):
			sys.exit("ERROR. 2 validators give diff results (1).")
		if errors and (result == "ok"):
			sys.exit("ERROR. 2 validators give diff results (2).")

	if 0:
		if "hidden_layer_sizes" in hyperparams:
			print("666666666666666")
			print(hyperparams["hidden_layer_sizes"])
			hyperparams["hidden_layer_sizes"] = list(hyperparams["hidden_layer_sizes"])
			print(hyperparams["hidden_layer_sizes"])
			print(type(hyperparams["hidden_layer_sizes"]))

		print(result)
		print(hyperparams)
		for error in errors:
			print(error)
			for suberror in sorted(error.context, key=lambda e: e.schema_path):
				print(list(suberror.absolute_schema_path))
		print(">_<")

	#print("Errors:",errors, "Result =",result)

	if 0:
		for error in errors:
			hp_that_appear = []
			ss = None
			#print("error:",error)
			#print("error.context:",error.context)
			for suberror in sorted(error.context, key=lambda e: e.schema_path):
				#print("suberror:",suberror)
				ss = list(suberror.absolute_schema_path)
				#print(ss)
				sanity_1 = 0
				hh = ""
				for s in ss:
					if s in all_hp:
						sanity_1 += 1
						hh = s
				if sanity_1 > 1 or hh == "":
					#print("sanity_1:", sanity_1)
					#print(hh)
					hh = "BUG in hp_that_appear"
					#sys.exit("ERROR. There should be only 1 hp in error.context[i].schema_path")

				if hh not in hp_that_appear:
					hp_that_appear.append(hh)

			for x in hp_that_appear:
				if x in "BUG in hp_that_appear":
					continue
				if schema_type == "hw":
					failed_hp_trials_hw_dict[x] += 1
				elif schema_type == "auto":
					failed_hp_trials_auto_dict[x] += 1
				elif schema_type == "wp":
					failed_hp_trials_wp_dict[x] += 1			

			#print(".. ", ss[1], len(schema[ss[0]]))
			# error.schema can be subschema so we need a bigger one aka constraint
			if ss is None:
				print("PANIC ???")
				const_sch = "Weird BUG in status_for_schema. PANIC????"
				print("schema_type =",schema_type)
				#assert False, "SSADASD"
			else:
				if ss[1] < len(schema[ss[0]]):
					#print(pprint.pformat(schema[ss[0]][ss[1]], width = 120, sort_dicts = False))
					const_sch = schema[ss[0]][ss[1]]
				else:
					sys.exit("ERROR. hmmmm")

			if schema_type == "hw":
				if const_sch not in failed_schema_hw:
					failed_schema_hw.append(const_sch)
				idx = str(failed_schema_hw.index(const_sch))
				if idx in failed_schema_hw_dict:
					failed_schema_hw_dict[idx] += 1
				else:
					failed_schema_hw_dict[idx] = 1
			elif schema_type == "auto":
				if const_sch not in failed_schema_auto:
					failed_schema_auto.append(const_sch)
				idx = str(failed_schema_auto.index(const_sch))
				if idx in failed_schema_auto_dict:
					failed_schema_auto_dict[idx] += 1
				else:
					failed_schema_auto_dict[idx] = 1
			elif schema_type == "wp":
				if const_sch not in failed_schema_wp:
					failed_schema_wp.append(const_sch)
				idx = str(failed_schema_wp.index(const_sch))
				if idx in failed_schema_wp_dict:
					failed_schema_wp_dict[idx] += 1
				else:
					failed_schema_wp_dict[idx] = 1




		#print(error.validator)
		#print(error.validator_value)
		#print(error.context)
		#print(error.schema)
		#for suberror in sorted(error.context, key=lambda e: e.schema_path):
		#	print(list(suberror.schema_path), suberror.message, sep=", ")
	#print("<<<<<<<<<<<")

	return result


def analyze_results(df, baseline, other):
    true_positives = len(df[(df[baseline]=="ok") & (df[other]=="ok")])
    false_positives = len(df[(df[baseline]!="ok") & (df[other]=="ok")])
    true_negatives = len(df[(df[baseline]!="ok") & (df[other]!="ok")])
    false_negatives = len(df[(df[baseline]=="ok") & (df[other]!="ok")])
    global_FP = false_positives
    global_TN = true_negatives
    if true_positives + false_positives == 0:
    	precision = "INF"
    else:
    	precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives == 0:
    	recall = "INF"
    else:
    	recall = true_positives / (true_positives + false_negatives)
    return [other.replace("status_", ""), true_positives, false_positives, true_negatives, false_negatives, precision, recall]

def f(x, skipUtilsValidation):
	failed_schema_hw.clear()
	failed_schema_auto.clear()
	failed_schema_wp.clear()
	failed_schema_hw_dict.clear()
	failed_schema_auto_dict.clear()
	failed_schema_wp_dict.clear()
	all_hp.clear()
	print_hw_const.clear()
	print_auto_const.clear()
	print_wp_const.clear()
	all_trials_hp.clear()
	hw_failed_trials_hp.clear()
	auto_failed_trials_hp.clear()
	wp_failed_trials_hp.clear()

	hw_TruePos.clear()
	hw_FalsePos.clear()
	hw_TrueNeg.clear()
	hw_FalseNeg.clear()
	auto_TruePos.clear()
	auto_FalsePos.clear()
	auto_TrueNeg.clear()
	auto_FalseNeg.clear()
	wp_TruePos.clear()
	wp_FalsePos.clear()
	wp_TrueNeg.clear()
	wp_FalseNeg.clear()

	op_handwritten = getlale.get_lale_handwritten(x)
	relevantToOptimizerHP = getlale.getRelevantToOptimizer(x)
	op_autogen = getlale.get_lale_autogen(x)
	assert (op_handwritten is not None) or (op_autogen is not None), "No HW and Autogen"
	
	op_base = op_handwritten
	if op_base is None:
		op_base = op_autogen

	op_base = op_base.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
	op_unconstrained = replace_constraints(op_base, [])

	print("relevantToOptimizerHP", relevantToOptimizerHP)

	if op_autogen is not None:
		op_docstrings = replace_constraints(op_base, get_constraints(op_autogen))
	else:
		op_docstrings = None
	JSS_WP_path = getlale.get_JSS_WP_path(x)
	data = []
	with open(JSS_WP_path, "r") as f:
		t = f.read()
# Subscript(value=Dict(keys=[Constant(value='XXX TODO XXX')], values=[Constant(value='_get_config()')]), slice=Constant(value='assume_finite'), ctx=Load())
		data = ast.literal_eval(t)
	with open("tt4WP.py", "w") as f:
		print(pprint.pformat(data, width = 200, sort_dicts = False), file = f)
	op_WPanalysis = replace_constraints(op_base, data) 

	if write_schema_to_file:
	#if True:
		f_write_schema_to_file(op_handwritten, op_unconstrained, op_docstrings, op_WPanalysis)

	#TODO: double check this
	print("BEFORE HYPEROPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
	extraList = ["MissingIndicator", "NMF", "VotingClassifier", "ColumnTransformer", "Pipeline", "RFE"
				,"VotingRegressor", "CalibratedClassifierCV"]
	multi_operators = ["MultiTaskElasticNet", "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV",]
	pipelinePos = 0
	newsgroups_train_XX = None
	newsgroups_train_yy = None

	if x in multi_operators:
		f_train_X = multi_data.train_X
		f_train_y = multi_data.train_y
		c_type = "multi"
		random_trainable = Hyperopt(
			estimator=op_unconstrained, max_evals=n_trials, algo="rand", cv=2, verbose = True)
		random_trained = random_trainable.fit(f_train_X, f_train_y)	
	elif x in extraList:
		c_type = "transformer"
		if x == "MissingIndicator":
			#temp = MissingIndicator(features="all")
			#temp = MissingIndicator(sparse=False)
			temp = MissingIndicator(features="all", sparse=False)
			temp = temp.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			trainable = (
				NoOp & (temp >> OrdinalEncoder()) #TODO freeze here too?
			) >> ConcatFeatures >> LogisticRegression().freeze_trainable()
			pipelinePos = 1
		elif x == "NMF":
			temp = NMF()
			temp = temp.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			trainable = (
				MinMaxScaler(feature_range=(0,1), clip=True)
				>> temp >> LogisticRegression().freeze_trainable()
			)
			pipelinePos = 1
		elif x == "VotingClassifier":
			trainable = VotingClassifier(
				estimators=[
					("lr", LogisticRegression().freeze_trainable()),
					("knn", KNeighborsClassifier().freeze_trainable()),
					("tree", DecisionTreeClassifier().freeze_trainable()),
					]
			)
			pipelinePos = 90
		elif x == "VotingRegressor":
			from lale.lib.sklearn import VotingRegressor, KNeighborsRegressor, DecisionTreeRegressor, LinearSVR
			trainable = VotingRegressor(
				estimators=[
					("lsvr", LinearSVR().freeze_trainable()),
					("knr", KNeighborsRegressor().freeze_trainable()),
					("treer", DecisionTreeRegressor().freeze_trainable()),
					]
			)
			trainable = trainable.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			pipelinePos = 90
		elif x == "ColumnTransformer":
			all_X, all_y = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
			all_columns = list(all_X.columns)
			temp = ColumnTransformer(
				transformers=[
					("pca", PCA().freeze_trainable(), all_columns),
					("passthrough", "passthrough", all_columns),
				], 
			)
			#relevantToOptimizerHP = ["sparse_threshold"]
			print(relevantToOptimizerHP)
			#fSad()
			temp = temp.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			#temp = temp.customize_schema(relevantToOptimizer=['sparse_threshold'])
			trainable = temp >> LogisticRegression(max_iter=1000).freeze_trainable()
			pipelinePos = 91
		elif x == "Pipeline":
			trainable = Pipeline(
				steps=[
					("pca", PCA()),
					("lr", LogisticRegression(max_iter=1000).freeze_trainable()),
				],
			)
			trainable = trainable.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			pipelinePos = 92
		elif x == "RFE":
			temp = RFE(estimator=LogisticRegression(), importance_getter="auto")
			temp = temp.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			trainable = (
				temp
				>> LogisticRegression().freeze_trainable()
			)
		elif x == "CalibratedClassifierCV":
			c_type = "classifier"
			from lale.lib.autogen import CalibratedClassifierCV
			from lale.lib.sklearn import LinearSVC
			trainable = CalibratedClassifierCV(base_estimator=LinearSVC(random_state=0).freeze_trainable())
			trainable = trainable.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
			# replace_constraints calls lale.operators.make_operator which probably reset everything
			# Autogen's CalibratedClassifierCV does not have a constraints so we can are safe(?) to not call replace_constraints. 
			#trainable = replace_constraints(trainable, [])

		#trained = trainable.fit(iris.train_X, iris.train_y)
		#predicted = trained.predict(iris.test_X)
		random_trainable = Hyperopt(
			estimator=trainable, max_evals=n_trials, algo="rand", cv=2, verbose = True)
		random_trained = random_trainable.fit(iris.train_X, iris.train_y)

	elif x == "BaggingClassifier":
		c_type = "classifier"
		trainable = BaggingClassifier(base_estimator=DecisionTreeClassifier())
		#trainable = BaggingClassifier(base_estimator=LogisticRegression())
		trainable = trainable.customize_schema(relevantToOptimizer=relevantToOptimizerHP)
		random_trainable = Hyperopt(
			estimator=trainable, max_evals=n_trials, algo="rand", cv=2, verbose = True)
		random_trained = random_trainable.fit(iris.train_X, iris.train_y)

	elif op_base.is_classifier():
		# classfier
		c_type = "classifier"
		#print("CCCCCCCC'")
		f_train_X = iris.train_X
		f_train_y = iris.train_y
		f_test_X = iris.test_X
		f_test_y = iris.test_y
		random_trainable = Hyperopt(
			estimator=op_unconstrained, max_evals=n_trials, algo="rand", cv=2, verbose = True)
		random_trained = random_trainable.fit(f_train_X, f_train_y)

	elif op_base.is_transformer():
		# transformer
		c_type = "transformer"

		if x == "TfidfVectorizer":
			categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
			newsgroups_train_X, newsgroups_train_y = sklearn.datasets.fetch_20newsgroups(subset='train', categories=categories, return_X_y= True)
			newsgroups_train_XX = newsgroups_train_X
			newsgroups_train_yy = newsgroups_train_y

			random_trainable = Hyperopt(
				estimator=op_unconstrained >> LR.freeze_trainable(), max_evals=n_trials, algo="rand", cv=2, verbose = True)
			print("AFTER HYPEROPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
			random_trained = random_trainable.fit(newsgroups_train_X, newsgroups_train_y)

		elif x == "StackingClassifier":
			trainable = StackingClassifier(estimators=[("base", LogisticRegression())])
			#trainable = StackingClassifier(estimators=[("base", LogisticRegression().freeze_trainable())])
			trainable = trainable.customize_schema(relevantToOptimizer=["cv", "passthrough"])
			trainable = replace_constraints(trainable, [])
			trainable = trainable(estimators=[("base", LogisticRegression())])
			
			with open("tt9ZZ.py", "w") as f:
				print(pprint.pformat(trainable.hyperparam_schema(), width = 200, sort_dicts = False), file = f)
				print(pprint.pformat(trainable._hyperparams, width = 200, sort_dicts = False), file = f)
				print(pprint.pformat(trainable.get_defaults(), width = 200, sort_dicts = False), file = f)
				print(pprint.pformat(trainable.hyperparams_all(), width = 200, sort_dicts = False), file = f)
				print(pprint.pformat(trainable._get_params_all(), width = 200, sort_dicts = False), file = f)	
				print(pprint.pformat(trainable._frozen_hyperparams, width = 200, sort_dicts = False), file = f)
				print(pprint.pformat(trainable._schemas, width = 200, sort_dicts = False), file = f)
			random_trainable = Hyperopt(
				estimator=trainable, max_evals=n_trials, algo="rand", cv=2, verbose = True)
			random_trained = random_trainable.fit(iris.train_X, iris.train_y)
			pipelinePos = 93

		elif x == "StackingRegressor":
			c_type = "regression"
			trainable = StackingRegressor(estimators=[("base", SGDRegressor())])
			#trainable = StackingClassifier(estimators=[("base", LogisticRegression().freeze_trainable())])
			trainable = trainable.customize_schema(relevantToOptimizer=["cv", "passthrough"])
			random_trainable = Hyperopt(
			    estimator=trainable, max_evals=n_trials, algo="rand", verbose = True, scoring="r2")
			random_trained = random_trainable.fit(regression_data.train_X, regression_data.train_y)

		else:
			random_trainable = Hyperopt(
				estimator=op_unconstrained >> LR.freeze_trainable(), max_evals=n_trials, algo="rand", cv=2, verbose = True)
			print("AFTER HYPEROPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
			random_trained = random_trainable.fit(iris.train_X, iris.train_y)

		
	else:
		# untagged operators
		untagged_classifiers = ["BernoulliNB", "Perceptron"]
		if x in untagged_classifiers:
			c_type = "classifier"
			f_train_X = iris.train_X
			f_train_y = iris.train_y
			random_trainable = Hyperopt(
				estimator=op_unconstrained, max_evals=n_trials, algo="rand", cv=2, verbose = True)
			random_trained = random_trainable.fit(f_train_X, f_train_y)

		else:
			# regression ???
			c_type = "regression"
			f_train_X = regression_data.train_X
			f_train_y = regression_data.train_y
			f_test_X = regression_data.test_X
			f_test_y = regression_data.test_y		

			random_trainable = Hyperopt(
			    estimator=op_unconstrained, max_evals=n_trials, algo="rand", verbose = True, scoring="r2")
			random_trained = random_trainable.fit(f_train_X, f_train_y)

	#random_trainable = Hyperopt(
	#    estimator=op_unconstrained, max_evals=n_trials, algo="rand", cv=2, verbose = True)
	#random_trained = random_trainable.fit(f_train_X, f_train_y)

	if c_type == "regression":
		f_train_X = regression_data.train_X
		f_train_y = regression_data.train_y
	else:
		f_train_X = iris.train_X
		f_train_y = iris.train_y

	if x == "TfidfVectorizer":
		f_train_X = newsgroups_train_yy
		f_train_y = newsgroups_train_XX

	global global_f_train_X
	global global_f_train_y
	global_f_train_X = f_train_X
	global_f_train_y = f_train_y

	if sanity_check:
		f_sanity_check(random_trained, x)

	# ----- validate configurations with handwritten schemas
	results = random_trained.summary()

	#print(random_trained.get_pipeline("p0").steps()[0].hyperparams())
	
	results["status_handwritten"] = results.index.map(
	    lambda name: status_for_schema(random_trained.get_pipeline(name), op_handwritten, 0, c_type, "hw", pipelinePos))
	results["status_docstrings"] = results.index.map(
	    lambda name: status_for_schema(random_trained.get_pipeline(name), op_docstrings, 0, c_type, "auto", pipelinePos))
	results["status_WPanalysis"] = results.index.map(
	    lambda name: status_for_schema(random_trained.get_pipeline(name), op_WPanalysis, 0, c_type, "wp", pipelinePos))
	#results["status_uncon"] = results.index.map(
	#    lambda name: status_for_schema(random_trained.get_pipeline(name), op_unconstrained))
	print(results)

	#assert False, "AAAAAAAAA"

	hw_TP = ""
	hw_FP = ""
	hw_TN = ""
	hw_FN = ""
	hw_FP_dict_counter = {}
	hw_TN_dict_counter = {}
	auto_TP = ""
	auto_FP = ""
	auto_TN = ""
	auto_FN = ""
	auto_FP_dict_counter = {}
	auto_TN_dict_counter = {}
	wp_TP = ""
	wp_FP = ""
	wp_TN = ""
	wp_FN = ""
	wp_FP_dict_counter = {}
	wp_TN_dict_counter = {}

	hw_FP_exception_stack = ""
	hw_TN_exception_stack = ""
	auto_FP_exception_stack = ""
	auto_TN_exception_stack = ""
	wp_FP_exception_stack = ""
	wp_TN_exception_stack = ""


	if op_handwritten is not None:
		result_HW = analyze_results(results, "status", "status_handwritten")
	else:
		result_HW = ["handwritten", "-", "-", "-", "-", "-", "-"]
	hw_FP_counter = result_HW[2]
	hw_TN_counter = result_HW[3]
	if op_docstrings is not None:
		result_Docstr = analyze_results(results, "status", "status_docstrings")
	else:
		result_Docstr = ["docstrings", "-", "-", "-", "-", "-", "-"]
	auto_FP_counter = result_Docstr[2]
	auto_TN_counter = result_Docstr[3]
	result_WP = analyze_results(results, "status", "status_WPanalysis")
	wp_FP_counter = result_WP[2]
	wp_TN_counter = result_WP[3]

	iii = 0

	for i, row in results.iterrows(): #slow but whatever
		#print(iii)
		hpp = str(all_trials_hp[iii])
		str_e = row["str_e"]

		if op_handwritten is not None:
			if row["status"] == "ok" and row["status_handwritten"] == "ok":
				hw_TP += hpp + "\n"
			elif row["status"] == "fail" and row["status_handwritten"] == "ok":
				hw_FP += hpp + "\n"
				hw_FP_exception_stack += row["error_msg"] + "\n" + str(hpp) + "\n===============================================================================\n"
				if str_e in hw_FP_dict_counter:
					hw_FP_dict_counter[str_e] += 1
				else:
					hw_FP_dict_counter[str_e] = 1
			elif row["status"] == "fail" and row["status_handwritten"] == "fail":
				hw_TN += hpp + "\n"
				hw_TN_exception_stack += row["error_msg"] + "\n" + str(hpp) + "\n===============================================================================\n"
				if str_e in hw_TN_dict_counter:
					hw_TN_dict_counter[str_e] += 1
				else:
					hw_TN_dict_counter[str_e] = 1
			elif row["status"] == "ok" and row["status_handwritten"] == "fail":
				hw_FN += hpp + "\n"

		if op_docstrings is not None:
			if row["status"] == "ok" and row["status_docstrings"] == "ok":
				auto_TP += hpp + "\n"
			elif row["status"] == "fail" and row["status_docstrings"] == "ok":
				auto_FP += hpp + "\n"
				auto_FP_exception_stack += row["error_msg"] + "\n" + str(hpp) + "\n===============================================================================\n"
				if str_e in auto_FP_dict_counter:
					auto_FP_dict_counter[str_e] += 1
				else:
					auto_FP_dict_counter[str_e] = 1
			elif row["status"] == "fail" and row["status_docstrings"] == "fail":
				auto_TN += hpp + "\n"
				auto_TN_exception_stack += row["error_msg"] + "\n" + str(hpp) + "\n===============================================================================\n"
				if str_e in auto_TN_dict_counter:
					auto_TN_dict_counter[str_e] += 1
				else:
					auto_TN_dict_counter[str_e] = 1
			elif row["status"] == "ok" and row["status_docstrings"] == "fail":
				auto_FN += hpp + "\n"

		if row["status"] == "ok" and row["status_WPanalysis"] == "ok":
			wp_TP += hpp + "\n"
		elif row["status"] == "fail" and row["status_WPanalysis"] == "ok":
			wp_FP += hpp + "\n"
			wp_FP_exception_stack += row["error_msg"] + "\n" + str(hpp) + "\n===============================================================================\n"
			if str_e in wp_FP_dict_counter:
				wp_FP_dict_counter[str_e] += 1
			else:
				wp_FP_dict_counter[str_e] = 1
		elif row["status"] == "fail" and row["status_WPanalysis"] == "fail":
			wp_TN += hpp + "\n"
			wp_TN_exception_stack += row["error_msg"] + "\n" + str(hpp) + "\n===============================================================================\n"
			if str_e in wp_TN_dict_counter:
				wp_TN_dict_counter[str_e] += 1
			else:
				wp_TN_dict_counter[str_e] = 1
		elif row["status"] == "ok" and row["status_WPanalysis"] == "fail":
			wp_FN += hpp + "\n"

		iii += 1

	#rr = analyze_results(results, "status", "status_uncon")
	#return [x, hw_total, hw_todo, auto_total, auto_todo, wp_total, wp_todo, iwp_total, iwp_todo]
	#		 0     1        2          3           4         5         6         7          8
	rr = checkConstNum.f(x)
	print(rr)

	if str(rr[1]) != "N/A":
		result_HW[0] = result_HW[0] + "_(" + str(rr[1] - rr[2]) + "/" + str(rr[1]) + ")"
	if str(rr[3]) != "N/A":
		result_Docstr[0] = result_Docstr[0] + "_(" + str(rr[3] - rr[4]) + "/" + str(rr[3]) + ")"
	result_WP[0] = result_WP[0] + "_(" + str(rr[7] - rr[8]) + "/" + str(rr[7]) + ")"

	num_ok = len(results[results["status"] == "ok"])
	num_fail = len(results[results["status"] !="ok"])


	if write_result_to_file:
		#(name)_0_result < include HP stats + wp(all) stats + failed const freq stats
		#(name)_1_hw_const
		#(name)_2_auto_const
		#(name)_3_wp_const
		#(name)_4_all_trials_hp
		#(name)_5_hw_failed_trials_hp
		#(name)_6_auto_failed_trials_hp
		#(name)_7_wp_failed_trials_hp
		#(name)_8_dumps

		if skipUtilsValidation:
			write_path = "\PATH_TO\result[skipValidation]"
			assert False
		else:
			write_path = "\PATH_TO\result"

		with open(write_path / (x + "_0_result"), "w") as f:
			print(">>>", x, " (", c_type, ")", file = f)
			print(num_fail + num_ok, "trials:", num_ok, "passed,", num_fail, "failed.", file = f)
			print(tabulate([result_HW, result_Docstr, result_WP], 
				headers = ["name", "true_positives", "false_positives", "true_negatives", "false_negatives", "precision", "recall"], 
				tablefmt = "pretty"), file = f)
			print("Note:", file = f)
			print("\thandwritten_(A/B) means there are B total constraints, A of which are good constraints (no TODO).", file = f)
			print("\tWP aboved are interesting wp, ie. 2+ hp or 1 + X/y. For overall WP, there are " + str(rr[5]) + " total, " + str(rr[5] - rr[6]) + " are good. All WP constraints are at /output/JSS_all_exceptions/", file = f)

			print("\n> Hyperparams: \n", all_hp, file = f)
			print("\n> relevantToOptimizer:\n",relevantToOptimizerHP, file = f)

			print("\n> handwritten", file = f)
			print("False Positives (" + str(hw_FP_counter) + "):", file = f)
			for k, v in sorted(hw_FP_dict_counter.items(), key=lambda x: x[1], reverse=True):
				print("\t(" + str(v) + ") " + str(k), file = f)
			print("True Negatives (" + str(hw_TN_counter) + "):", file = f)	
			for k, v in sorted(hw_TN_dict_counter.items(), key=lambda x: x[1], reverse=True):
				print("\t(" + str(v) + ") " + str(k), file = f)

			print("\n> docstring", file = f)
			print("False Positives (" + str(auto_FP_counter) + "):", file = f)
			for k, v in sorted(auto_FP_dict_counter.items(), key=lambda x: x[1], reverse=True):
				print("\t(" + str(v) + ") " + str(k), file = f)
			print("True Negatives (" + str(auto_TN_counter) + "):", file = f)	
			for k, v in sorted(auto_TN_dict_counter.items(), key=lambda x: x[1], reverse=True):
				print("\t(" + str(v) + ") " + str(k), file = f)

			print("\n> WPanalysis", file = f)
			print("False Positives (" + str(wp_FP_counter) + "):", file = f)
			for k, v in sorted(wp_FP_dict_counter.items(), key=lambda x: x[1], reverse=True):
				print("\t(" + str(v) + ") " + str(k), file = f)
			print("True Negatives (" + str(wp_TN_counter) + "):", file = f)	
			for k, v in sorted(wp_TN_dict_counter.items(), key=lambda x: x[1], reverse=True):
				print("\t(" + str(v) + ") " + str(k), file = f)


			print("\n> # of time they appear in failed validations:", file = f)
			print("handwritten: \n", failed_hp_trials_hw_dict, file = f)
			print("docstring: \n", failed_hp_trials_auto_dict, file = f)
			print("WPanalysis: \n", failed_hp_trials_wp_dict, file = f)


			print("\n> Failed constraints stats:", file = f)
			print("handwritten:", file = f)
			for w in failed_schema_hw:
				idx = str(failed_schema_hw.index(w))
				print("freq =", failed_schema_hw_dict[idx], file = f)
				print(pprint.pformat(w, width = 120, sort_dicts = False), file = f)

			print("\ndocstring:", file = f)
			for w in failed_schema_auto:
				idx = str(failed_schema_auto.index(w))
				print("freq =", failed_schema_auto_dict[idx], file = f)
				print(pprint.pformat(w, width = 120, sort_dicts = False), file = f)

			print("\nWPanalysis:", file = f)
			for w in failed_schema_wp:
				idx = str(failed_schema_wp.index(w))
				print("freq =", failed_schema_wp_dict[idx], file = f)
				print(pprint.pformat(w, width = 120, sort_dicts = False), file = f)

		write_path_X = "PATH_TO\failed_WP_schema"
		with open(write_path_X / (x + "_failed_WP_schema"), "w") as f:
			for w in failed_schema_wp:
				idx = str(failed_schema_wp.index(w))
				print(pprint.pformat(w, width = 120, sort_dicts = False), file = f)		

		with open(write_path / (x + "_1_hw_const"), "w") as f:
			print(pprint.pformat(print_hw_const, width = 120, sort_dicts = False), file = f)
		with open(write_path / (x + "_2_auto_const"), "w") as f:
			print(pprint.pformat(print_auto_const, width = 120, sort_dicts = False), file = f)
		with open(write_path / (x + "_3_wp_const"), "w") as f:
			print(pprint.pformat(print_wp_const, width = 1200, sort_dicts = False), file = f)

		with open(write_path / (x + "_4_all_trials_hp"), "w") as f:	
			print(pprint.pformat(all_trials_hp, width = 120, sort_dicts = False), file = f)

		if 0:
			with open(write_path / (x + "_5_hw_failed_trials_hp"), "w") as f:	
				print(pprint.pformat(hw_failed_trials_hp, width = 120, sort_dicts = False), file = f)
			with open(write_path / (x + "_6_auto_failed_trials_hp"), "w") as f:	
				print(pprint.pformat(auto_failed_trials_hp, width = 120, sort_dicts = False), file = f)
			with open(write_path / (x + "_7_wp_failed_trials_hp"), "w") as f:	
				print(pprint.pformat(wp_failed_trials_hp, width = 120, sort_dicts = False), file = f)

		with open(write_path / (x + "_8_dumps"), "w") as f:	
			ddd = [num_ok, num_fail, result_HW, result_Docstr, result_WP, all_hp, failed_hp_trials_hw_dict, failed_hp_trials_auto_dict, 
					failed_hp_trials_wp_dict, failed_schema_hw, failed_schema_hw_dict, failed_schema_auto, failed_schema_auto_dict, 
					failed_schema_wp, failed_schema_wp_dict]
			print(pprint.pformat(ddd, width = 120, sort_dicts = False), file = f)
		if skipUtilsValidation:
			write_path_2 = "PATH_TO\TrueFalsePosNeg[skipValidation]"
			assert False
		else:
			write_path_2 = "PATH_TO\TrueFalsePosNeg"
		with open(write_path_2 / (x + "_1a_hw_TruePos"), "w") as f:
			#print(pprint.pformat(hw_TP, width = 120, sort_dicts = False), file = f)
			print(hw_TP, file = f)
		with open(write_path_2 / (x + "_1b_hw_FalsePos"), "w") as f:
			print(hw_FP, file = f)
		with open(write_path_2 / (x + "_1c_hw_TrueNeg"), "w") as f:
			print(hw_TN, file = f)
		with open(write_path_2 / (x + "_1d_hw_FalseNeg"), "w") as f:
			print(hw_FN, file = f)

		with open(write_path_2 / (x + "_2a_auto_TruePos"), "w") as f:
			print(auto_TP, file = f)
		with open(write_path_2 / (x + "_2b_auto_FalsePos"), "w") as f:
			print(auto_FP, file = f)
		with open(write_path_2 / (x + "_2c_auto_TrueNeg"), "w") as f:
			print(auto_TN, file = f)
		with open(write_path_2 / (x + "_2d_auto_FalseNeg"), "w") as f:
			print(auto_FN, file = f)

		with open(write_path_2 / (x + "_3a_wp_TruePos"), "w") as f:
			print(wp_TP, file = f)
		with open(write_path_2 / (x + "_3b_wp_FalsePos"), "w") as f:
			print(wp_FP, file = f)
		with open(write_path_2 / (x + "_3c_wp_TrueNeg"), "w") as f:
			print(wp_TN, file = f)
		with open(write_path_2 / (x + "_3d_wp_FalseNeg"), "w") as f:
			print(wp_FN, file = f)

		with open(write_path_2 / (x + "_4a_hw_FP_exception_stack"), "w") as f:
			print(hw_FP_exception_stack, file = f)
		with open(write_path_2 / (x + "_4b_hw_TN_exception_stack"), "w") as f:
			print(hw_TN_exception_stack, file = f)
		with open(write_path_2 / (x + "_5a_auto_FP_exception_stack"), "w") as f:
			print(auto_FP_exception_stack, file = f)
		with open(write_path_2 / (x + "_5b_auto_TN_exception_stack"), "w") as f:
			print(auto_TN_exception_stack, file = f)
		with open(write_path_2 / (x + "_6a_wp_FP_exception_stack"), "w") as f:
			print(wp_FP_exception_stack, file = f)
		with open(write_path_2 / (x + "_6b_wp_TN_exception_stack"), "w") as f:
			print(wp_TN_exception_stack, file = f)

		write_path_ABC = "PATH_TO\result"
		with open(write_path_ABC / "add_HW_dense_stats.py","a+") as f:
			ooo = [x,result_HW[1],result_HW[2],result_HW[3],result_HW[4]]
			print(ooo, file = f)

		with open(write_path_ABC / "add_Auto_dense_stats.py","a+") as f:
			ooo = [x,result_Docstr[1],result_Docstr[2],result_Docstr[3],result_Docstr[4]]
			print(ooo, file = f)
		
		with open(write_path_ABC / "add_WP_dense_stats.py","a+") as f:
			ooo = [x,result_WP[1],result_WP[2],result_WP[3],result_WP[4]]
			print(ooo, file = f)

		if 0:
			with open(write_path_precRecall / "prec_recall_dump.py","w") as f:
				if op_docstrings is None:
					o = "N/A"
				else:
					o = result_Docstr[0].split("_")[1]
				oo = result_HW[0].split("_")[1] + " " + o + " " + result_WP[0].split("_")[1]
				ooo = [x,result_HW[5],result_HW[6],result_Docstr[5],result_Docstr[6],result_WP[5],result_WP[6],oo]
				print(ooo, file = f)

		#with open(write_path_3 / x, "w") as f:
		#	print(pprint.pformat(hp_distinction, width = 120, sort_dicts = False), file = f)


	print(tabulate([result_HW, result_Docstr, result_WP], 
		headers = ["name", "true_positives", "false_positives", "true_negatives", "false_negatives", "precision", "recall"], 
		tablefmt = "pretty"))

	print("All hp:")
	print(all_hp)
	
	print("Failed schema hw:")
	print(pprint.pformat(failed_schema_hw, width = 120, sort_dicts = False))
	print(failed_schema_hw_dict)
	print(failed_hp_trials_hw_dict)
	
	print("Failed schema auto:")
	print(failed_schema_auto)

	print("Failed schema wp:")
	print(pprint.pformat(failed_schema_wp, width = 120, sort_dicts = False))
	print(failed_schema_wp_dict)
	print(failed_hp_trials_wp_dict)



	print("END ", x, "\n\n")




#DOWNN
n_trials = 1000
write_schema_to_file = False
sanity_check = False
write_result_to_file = True

f("PCA", False)


# We can just run f() on "ops" but we do it separately just for sanity check + easier to do hand-inspection.
# print("WARNING!!! get_JSS_WP_path only return /inter_result/ALL_testing_inter_PCA_JSS.py")

        

# is_classifier(), is_transformer()
# is_supervised(), is_frozen_trainable(), is_frozen_trained(), is_relevant()
