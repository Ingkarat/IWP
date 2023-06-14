import ast
from equality import Impl

class Assign_Analyzer(ast.NodeVisitor):
	def __init__(self):
		self.modified_vars = []

	def visit_FunctionDef(self, node):
		super(Assign_Analyzer, self).generic_visit(node)

	def visit_Return(self, node):
		#print(">>> At visit_Return")
		#print(ast.dump(node))
		super(Assign_Analyzer, self).generic_visit(node)

	def addTo(self, node):
		if isinstance(node,ast.Subscript):
			self.addTo(node.value)
		if isinstance(node,ast.Name):
			if node.id not in self.modified_vars:
				self.modified_vars.append(node.id)
		if isinstance(node,ast.Attribute):
			#print(ast.dump(node))
			if node.value.id == "self":
				if node.attr not in self.modified_vars:
					self.modified_vars.append(node.attr)
			else:
				self.addTo(node.value)	
		if isinstance(node,ast.Tuple):
			for elt in node.elts:
				self.addTo(elt)
			
	def visit_Assign(self, node):
		#print(">>> At visit_Assign")
		#print(ast.dump(node))
		#print(ast.unparse(node))

		# target is a single node and can be a Name, a Attribute or a Subscript
		for target in node.targets:
			self.addTo(target)
		super(Assign_Analyzer, self).generic_visit(node)

	def visit_AugAssign(self, node):
		self.addTo(node.target)
		super(Assign_Analyzer, self).generic_visit(node)	


def getArgs(node):
	# posonlyargs: no function has this
	params = []
	#print(ast.dump(node))
	if node.args:
		for arg in node.args:
			if arg.arg != "self":
				params.append(arg.arg)

	if node.kwonlyargs:
		for kw in node.kwonlyargs:
			params.append(kw.arg)
		i = 1

	# vararg, kwarg
	# defaults, kw_defaults

	#print(params)
	return params


# Check if return args is a subset of function's params
# Check if all return args are unmodified
#
# function's params: take from args and kwonlyargs
# return args: take from ast.Return.Name, ast.Return.Tuple
#
# UNSOUND
def checkSpecialFunction(function_map):
	result = {}

	for key in function_map.keys():
		node = function_map[key]
		params = getArgs(node.args)
		returns = []
		if node.body:
			for b in node.body:
				if isinstance(b,ast.Return):
					#print(key)
					#print(ast.dump(b))
					#print(">>>", ast.unparse(b))
					if isinstance(b.value,ast.Name):
						returns.append(b.value.id)
					if isinstance(b.value,ast.Tuple):
						for v in b.value.elts:
							if isinstance(v,ast.Name):
								returns.append(v.id)
		
		if "self" in returns:
			returns.remove("self")

		#print("Return =",returns)

		if set(returns).issubset(set(params)) and returns and params:
			assign_ana = Assign_Analyzer()
			assign_ana.visit(node)
			mod_vars = assign_ana.modified_vars

			# Check that all params are not in mod_vars
			good = True
			for p in returns:
				if p in mod_vars:
					good = False
			if good:
				#result[key] = returns
				if len(returns) != 1:
					# Not support (for now) when returns have more than one value
					continue
				
					if 0:
						print(key)
						print(ast.dump(node))
						print("\n\n")
						print(returns)
					
					#imblearn
					if returns[0] == "X" and returns[1] == "y" and node.name == "_identity":
						continue

					assert False, "Case when returns has more than one value."
				if len(returns) == 1:
					r = returns[0]
					pos = -1
					current_pos = 0
					if node.args.args:
						for arg in node.args.args:
							if arg.arg != "self":
								if r == arg.arg:
									pos = current_pos
								current_pos += 1
					if pos != -1:
						result[key] = ["arg", pos]
					else:
						current_pos = 0
						if node.args.kwonlyargs:
							for kw in node.args.kwonlyargs:
								if r == kw.arg:
									pos = current_pos
								current_pos += 1
						result[key] = ["kwonlyarg", pos]
					assert pos != -1, "BUG"	

				if 0:
					print("\n\n")
					print(key)
					print(ast.unparse(node))
					print(params)
					print(returns)
					print(mod_vars)
					print(result[key])

	#print(result)
	return result

# node = ast.Call
# Check if this node is a special function call
def checkIfSpecial(node, helper_function):
	if 0:
		print("\n===\n")
		print(ast.dump(node))
		print("\n")
		print(ast.unparse(node))
		
	if isinstance(node.func, ast.Attribute):
		for key in helper_function.keys():
			if ";" in key:
				fname = key.split(";")[-1]
			else:
				fname = key.split(":")[-1]
			#if node.func.attr == key.split(":")[-1]:
			if node.func.attr == fname:
				#print(ast.dump(node))
				#print(ast.unparse(node))
				#print(key)
				try:
					if hasattr(node.func.value, "value"): # eg. self.le_.transform(y)
						return False
					if node.func.value.id == "estimator" and node.func.attr == "predict": # AdaBoostClassifier
						return False
					if node.func.value.id == "self" and node.func.attr == "predict":
						return False
				except:
					return False # for now

				return False #TODO: remove this

				print("---\n",print(ast.dump(node)))
				print(ast.unparse(node))
				assert False, "There is a special function with Attribute call aaa.bbb(ccc)"

	elif isinstance(node.func, ast.Name):
		for key in helper_function.keys():
			if ";" in key:
				fname = key.split(";")[-1]
			else:
				fname = key.split(":")[-1]
			#if node.func.id == key.split(":")[-1]:
			if node.func.id == fname:
				return True
	elif isinstance(node.func, ast.Call):
		# TODO: what about this case? For now it's irrelevant
		i = 1
	else:
		#SPLITTERS[self.splitter](criterion, self.max_features_, min_samples_leaf, min_weight_leaf, random_state)
		#(ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
		#assert False
		i = 1

	return False


# node = ast.Call
# Check if this node is a special function call
def getSpecialReturn(node, helper_function):
	print("\nAt getSpecialReturn\n")
	print(ast.dump(node))
	print(ast.unparse(node))

	r = ""
	for key in helper_function.keys():
		if ";" in key:
			fname = key.split(";")[-1]
		else:
			fname = key.split(":")[-1]
		if node.func.id == fname:
			r = helper_function[key]
	print(r)
	if r[0] == "arg":
		#check for "self"
		sself = 0
		if isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
			sself = 1
		print(ast.dump(node.args[r[1] + sself]))
		return node.args[r[1] + sself]
	elif r[0] == "kwonlyarg":
		return node.keywords[r[1]]
	assert False


def ToAst(x):
	if isinstance(x, ast.AST):
		return x
	elif isinstance(x, str):
		return ast.Name(id = x)

	return ast.Constant(value=x)

def ToStr(x):
	if not isinstance(x, ast.AST):
		return str(x)

	if isinstance(x, ast.Name):
		return str(x.id)

	print("\n", ast.dump(x))
	assert False, "helper.ToStr"

def checkIfRHSDict(x):
	if isinstance(x, ast.Call):
		if isinstance(x.func, ast.Name):
			if x.func.id == "dict":
				return True
	return False

def dictToAstDict(x):
	rep_dict = ast.Dict(keys=[], values=[])
	for k in x.keywords:
		rep_dict.keys.append(ast.Constant(value=k.arg))
		rep_dict.values.append(k.value)
	return rep_dict

def impl_counter(wp):
  cc = 0
  if isinstance(wp,Impl):
    return 1
  else:
    if wp is None:
      return cc
    for attr in wp.__dict__.keys():
      if isinstance(wp.__dict__[attr],ast.AST):
        cc += impl_counter(wp.__dict__[attr])
      elif isinstance(wp.__dict__[attr],list):
        for elt in wp.__dict__[attr]:
          cc += impl_counter(elt)
  return cc