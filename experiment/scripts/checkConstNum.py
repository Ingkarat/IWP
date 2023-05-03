import getlale
import lale.operators
import pprint
import ast
from tabulate import tabulate

def get_constraints(op):
    result = op.hyperparam_schema()["allOf"][1:]
    return result

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

otherType = []
def hasTODO(dd):
	if isinstance(dd, dict):
		for k, v in dd.items():
			if hasTODO(k):
				return True
			if hasTODO(v):
				return True
	elif isinstance(dd, list):
		for d in dd:
			if hasTODO(d):
				return True
	else:
		#print(dd)
		if isinstance(dd, str):
			if "XXX TODO XXX" in dd:
				return True
		else:
			# for sanity check
			if type(dd) not in otherType:
				otherType.append(type(dd))
	return False

def checkDict(dd):
	num = len(dd)
	hasTD = 0
	for d in dd:
		if hasTODO(d):
			hasTD += 1

	return num, hasTD

def f(x):
	op_handwritten = getlale.get_lale_handwritten(x)
	op_autogen = getlale.get_lale_autogen(x)

	op_base = op_handwritten
	if op_base is None:
		op_base = op_autogen

	op_unconstrained = replace_constraints(op_base, [])
	if op_autogen is not None:
		op_docstrings = replace_constraints(op_base, get_constraints(op_autogen))
	else:
		op_docstrings = None

	JSS_WP_path = getlale.get_JSS_WP_path(x)
	data = []
	with open(JSS_WP_path, "r") as f:
		t = f.read()
		data = ast.literal_eval(t)
	con_wp = data

	JSS_ALL_WP_path = getlale.get_JSS_ALL_WP_path(x)
	data_all = []
	with open(JSS_ALL_WP_path, "r") as f:
		t = f.read()
		data_all = ast.literal_eval(t)
	con_wp_all = data_all

	if op_handwritten is not None:
		con_hw = get_constraints(op_handwritten)
	if op_autogen is not None:
		con_auto = get_constraints(op_autogen)
	

	if 0:
		with open("zzz.py", "w") as f:
			print(pprint.pformat(con_hw, width = 200, sort_dicts = False), file = f)
			print("\n\n\n", file = f)
			print(pprint.pformat(con_auto, width = 200, sort_dicts = False), file = f)
			print("\n\n\n", file = f)
			print(pprint.pformat(con_wp, width = 200, sort_dicts = False), file = f)

	if op_handwritten is not None:		
	#if con_hw:
		hw_total, hw_todo = checkDict(con_hw)
		#print(hw_total, hw_todo)
	else:
		#hw_total = 0
		#hw_todo = 0
		hw_total = "N/A"
		hw_todo = "N/A"

	if op_autogen is not None:
		auto_total, auto_todo = checkDict(con_auto)
		#print(auto_total, auto_todo)
	else:
		auto_total = "N/A"
		auto_todo = "N/A"

	# check if WP const is not empty
	if con_wp:
		iwp_total, iwp_todo = checkDict(con_wp)
		#print(iwp_total, iwp_todo)
	else:
		iwp_total = 0
		iwp_todo = 0

	if con_wp_all:
		wp_total, wp_todo = checkDict(con_wp_all)
		#print(wp_total, wp_todo)
	else:
		wp_total = 0
		wp_todo = 0

	print("Other types:", otherType)
	#print(len(con_auto))
	#print(len(con_wp))

	return [x, hw_total, hw_todo, auto_total, auto_todo, wp_total, wp_todo, iwp_total, iwp_todo]

if 0:
	ops = ["AdaBoostClassifier", "AdaBoostRegressor", "BaggingClassifier", "ColumnTransformer", "DecisionTreeClassifier", "DecisionTreeRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "FeatureAgglomeration", "FunctionTransformer", "GaussianNB", "GradientBoostingClassifier", "GradientBoostingRegressor", "KNeighborsClassifier", "KNeighborsRegressor", "LinearRegression", "LinearSVC", "LogisticRegression", "MinMaxScaler", "MissingIndicator", "MLPClassifier", "MultinomialNB", "NMF", "Normalizer", "Nystroem", "OneHotEncoder", "OrdinalEncoder", "PassiveAggressiveClassifier", "PCA", "Pipeline", "PolynomialFeatures", "QuadraticDiscriminantAnalysis", "QuantileTransformer", "RandomForestClassifier", "RandomForestRegressor", "RFE", "Ridge", "RidgeClassifier", "RobustScaler", "SelectKBest", "SGDClassifier", "SGDRegressor", "SimpleImputer", "StandardScaler", "SVC", "SVR", "TfidfVectorizer", "VotingClassifier"]
	no_autogen = ["BaggingClassifier", "ColumnTransformer", "FeatureAgglomeration", "Pipeline", "RFE", "SelectKBest", "TfidfVectorizer", "VotingClassifier"]
	#currentlyExclude = ["DecisionTreeClassifier", "DecisionTreeRegressor", "OneHotEncoder", "OrdinalEncoder", "QuadraticDiscriminantAnalysis", "SelectKBest", "TfidfVectorizer"]
	currentlyExclude = ["OneHotEncoder", "OrdinalEncoder"]

	R = []
	for op in ops:
		if op not in currentlyExclude:
			print("Working on:", op)
			r = f(op)
			R.append(r)
	with open("constraintStat_7", "w") as f:
		print(tabulate(R, 
			headers = ["handwritten", "HW with TODO", "autogen", "auto with TODO", "WP(All)", "WP(All) with TODO", "Interested WP", "IWP with TODO"], 
			tablefmt = "pretty"), file = f)

	print("Sanity check. Types = ", otherType)

	# Number of constraints
	#handwritten, #HW with TODO, #autogen, #auto with TODO, #WeakestPrecon(All), #WP(All) with TODO, #WP that we talked about, #WP(.) with TODO



	print("555")


#f("LogisticRegression")