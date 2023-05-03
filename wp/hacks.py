import ast

def ins(obj,cls):
  return isinstance(obj,cls);

TRUE = ast.Constant(value=True,kind=None);
FALSE = ast.Constant(value=False,kind=None);

safe_funcs = ['check_is_fitted', '_check_sample_weight', 'check_consistent_length', 'check_classification_targets', 'warnings']

def hack(node):
  # {}.pop(arg1,arg2) simplifies to arg2 ... TODO: have to check whether Dict is empty.
  if ins(node,ast.Call) and ins(node.func,ast.Attribute) and node.func.attr == 'pop' and ins(node.func.value,ast.Dict) and len(node.args)==2:
    return node.args[1];
  # callable(ast.Consant) = FALSE; ugly that can't have simplify's FALSE
  elif is_func(node,'callable') and len(node.args)==1 and ins(node.args[0],ast.Constant):
    return FALSE;
  elif is_func(node,'isinstance') and len(node.args)==2:
    arg1 = node.args[0];
    arg2 = node.args[1];
    #print("I AM HERE!!! ",ast.dump(arg1),ast.dump(arg2))
    if ins(arg1,ast.Attribute) and (arg1.attr == 'float64') or ins(arg1,ast.Constant) and arg1.value=='None':
      return FALSE;

    if ins(arg1,ast.List) and (ins(arg2,ast.List) or ins(arg2,ast.Tuple)) and contains_list(arg2.elts):
      return TRUE;
    if ins(arg1,ast.Call) and ins(arg1.func,ast.Name) and ((arg1.func.id == 'min') or (arg1.func.id == 'max')) and ins(arg2,ast.Attribute) and arg2.attr == 'Number':
      return TRUE;
    if ins(arg1,ast.BinOp) and ins(arg2,ast.Attribute) and arg2.attr == 'Number':
      return TRUE;
    if ins(arg1,ast.Call) and ins(arg1.func,ast.Name) and ((arg1.func.id == 'min') or (arg1.func.id == 'max')) and ins(arg2,ast.Name) and arg2.id == 'str':
      return FALSE;
    if ins(arg1,ast.BinOp) and ins(arg2,ast.Name) and arg2.id == 'str':
      return FALSE;
  # [X,y][0] -> X and [X,y][1] -> y
  elif ins(node,ast.Subscript) and ins(node.value,ast.Tuple) and ins(node.slice,ast.Index) and ins(node.slice.value,ast.Constant):
    return node.value.elts[node.slice.value.value]
  return node;

def is_func(node,name):
  if ins(node,ast.Call) and ins(node.func,ast.Name) and (node.func.id == name):
    return True;

def is_safe_func(node):
  if ins(node, ast.Name) and node.id in safe_funcs:
    return True;
  else:
    return False; 

def contains_list(elts):
  for elem in elts:
    if ins(elem,ast.Name) and elem.id == 'list':
      return True
  else:
    return False   
    
