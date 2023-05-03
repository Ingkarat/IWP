import ast

# All AST Nodes that make up formulas
# Base operands: ast.Name, ast.Constant
# Composites operands: ast.Tuple, ast.Attribute, ast.List, ast.Tuple, ast.Subscript, ast.Index, ast.Call
# Operators: ast.UnaryOp, ast.Compare, ast.BinOp, ast.BoolOp, 

# ast.Add, ast.Sub, ast.Mult, ast.Div, ast.And, ast.Or, ast.Not, ast.Eq,
# ast.Not, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,  

class Impl(ast.AST):
  def __init__(self):
    name = 'Impl';
  
def ins(obj,cls):
  return isinstance(obj,cls);

def compare_Constant(node, other):
  if ins(other,ast.Constant):
    return node.value == other.value;
  else:
    return False;

def compare_Name(node,other):
  if ins(other,ast.Name):
    return node.id == other.id;
  else:
    return False;

def compare_Attribute(node,other):
  if ins(other,ast.Attribute):
    return compare(node.value,other.value) and node.attr == other.attr;
  else:
    return False;

def compare_Tuple(node,other):
  if ins(other,ast.Tuple):
    num = 0;
    for elem in node.elts:
      if (not compare(elem,other.elts[num])):
        return False;
      num+=1;
    return True;
  else:
    return False;

def compare_List(node,other):
  if ins(other,ast.List):
    num= 0;
    for elem in node.elts:
      if (not compare(elem,other.elts[num])):
          return False;
      num+=1;
    return True;
  else:
    return False;

def compare_Dict(node,other):
  if ins(other,ast.Dict):
    return compare(node.keys, other.keys) and compare(node.values,other.values);
  else:
    return False;

def compare_Call(node,other):
  if ins(other,ast.Call):
    if (not compare(node.func,other.func)): return False;
    num = 0;
    # What if len() is not equal? = always False?
    if len(node.args) > len(other.args):
      return False
    for arg in node.args:
      #print(">> ",num, len(node.args), len(other.args))
      if (not compare(arg,other.args[num])):
        return False;
      num+=1;
    return True;
  else:
    return False;

def compare_Subscript(node,other):
  if ins(other,ast.Subscript):
    return compare(node.value,other.value) and compare(node.slice,other.slice);
  else:
    return False;

def compare_Index(node,other):
  if ins(other,ast.Index):
    return compare(node.value,other.value);
  else:
    return False;

# Operators: ast.UnaryOp, ast.Compare, ast.BinOp, ast.BoolOp,

def compare_UnaryOp(node,other):
  if ins(other,ast.UnaryOp):
    return compare(node.op,other.op) and compare(node.operand,other.operand);
  else:
    return False;

def compare_BinOp(node,other):
  if ins(other,ast.BinOp):
    return compare(node.op,other.op) and compare(node.left,other.left) and compare(node.right,other.right);
  else:
    return False;

def compare_Compare(node,other):
  if ins(other,ast.Compare):
    if (not compare(node.left,other.left)): return False;
    num = 0;
    for op in node.ops:
      if (not compare(op,other.ops[num])):
          return False;
      num+=1;
    num = 0;
    for comp in node.comparators:
      if (not compare(comp,other.comparators[num])):
          return False;
      num+=1;
    return True;
  else:
    return False;

def compare_BoolOp(node,other):
  if ins(other,ast.BoolOp):
    if (not compare(node.op,other.op)): return False;
    num = 0;
    for value in node.values:
      if (not compare(value,other.values[num])):
          return False;
      num+=1;
    return True;
  else:
    return False;

# Operators!

def compare_Add(node,other):
  return ins(other,ast.Add);

def compare_Sub(node,other):
  return ins(other,ast.Sub);

def compare_Mult(node,other):
  return ins(other,ast.Mult);

def compare_Div(node,other):
  return ins(other,ast.Div);

def compare_FloorDiv(node,other):
  return ins(other,ast.FloorDiv);

def compare_BitOr(node,other):
  return ins(other,ast.BitOr);

def compare_BitXor(node,other):
  return ins(other,ast.BitXor);

def compare_BitAnd(node,other):
  return ins(other,ast.BitAnd);

def compare_And(node,other):
  return ins(other,ast.And);

def compare_Or(node,other):
  return ins(other,ast.Or);

def compare_Impl(node,other):
  return ins(other,Impl);

def compare_Not(node,other):
  return ins(other,ast.Not);

def compare_USub(node,other):
  return ins(other,ast.USub);

def compare_Invert(node, other):
  return ins(other, ast.Invert)

def compare_Eq(node,other):
  return ins(other,ast.Eq);

def compare_NotEq(node,other):
  return ins(other,ast.NotEq);

def compare_Lt(node,other):
  return ins(other,ast.Lt);

def compare_LtE(node,other):
  return ins(other,ast.LtE);

def compare_Gt(node,other):
  return ins(other,ast.Gt);

def compare_GtE(node,other):
  return ins(other,ast.GtE);

def compare_Is(node,other):
  return ins(other,ast.Is);

def compare_IsNot(node,other):
  return ins(other,ast.IsNot);

def compare_In(node,other):
  return ins(other,ast.In);

def compare_NotIn(node,other):
  return ins(other,ast.NotIn);

# Certain expressions. May occur as Base operands. TODO: Double check.

# def compare_GeneratorExp(node,other):
#  return ins(other, ast.GeneratorExp)


# TODO: ExtSlice has been deprecated
# def compare_ExtSlice(node, other):
#  return False 

def compare_IfExp(node, other):
  if ins(other, ast.IfExp):
   return compare(node.test,other.test) and compare(node.body,other.body) and compare(node.orelse,other.orelse)
  else:
   return False

# TODO: handle comparison of Slice expressions more intelligently
# def compare_Slice(node, other):
  #if ins(other, ast.Slice):
  #  return compare(node.lower,other.lower) and compare(node.upper,other.upper) and compare(node.step,other.step)
  #else:
#  return False

# TODO: handle comparisons better
# def compare_GeneratorExp(node, other):
#  return False

#def compare_Slice(node, other):
#  return ins(other, ast.Slice)

# Base operands: ast.Name, ast.Constant                                                  
# Composites operands: ast.Tuple, ast.Attribute, ast.List, ast.Subscript, ast.Index                                              
# Operators: ast.UnaryOp, ast.Compare, ast.BinOp, ast.BoolOp,                                                                      
# ast.Add, ast.Sub, ast.Mult, ast.Div, ast.And, ast.Or, ast.Not, ast.Eq,                                                               
# ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,  

def compare(node, other):
  if ins(node,ast.Name): 
    return compare_Name(node,other);
  elif ins(node,ast.Constant): 
    return compare_Constant(node,other);
  elif ins(node,ast.Attribute): 
    return compare_Attribute(node,other);
  elif ins(node,ast.Tuple): 
    return compare_Tuple(node,other);
  elif ins(node,ast.List): 
    return compare_List(node,other);
  elif ins(node,ast.Dict):
    return compare_Dict(node,other);
  elif ins(node,ast.Call):
    return compare_Call(node,other);
  elif ins(node,ast.Subscript): 
    return compare_Subscript(node,other);
  elif ins(node,ast.Index): 
    return compare_Index(node,other);
  # Operators: ast.UnaryOp, ast.Compare, ast.BinOp, ast.BoolOp, 
  elif ins(node,ast.UnaryOp):
    return compare_UnaryOp(node,other);
  elif ins(node,ast.Compare):
    return compare_Compare(node,other);
  elif ins(node,ast.BinOp):
    return compare_BinOp(node,other);
  elif ins(node,ast.BoolOp):
    return compare_BoolOp(node,other);
  # ast.Add, ast.Sub, ast.Mult, ast.Div, ast.And, ast.Or, ast.Not, ast.Eq,                                                                                  
  # ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
  elif ins(node,ast.Add):
    return compare_Add(node,other);
  elif ins(node,ast.Sub):
    return compare_Sub(node,other);
  elif ins(node,ast.Mult):
    return compare_Mult(node,other);
  elif ins(node,ast.Div):
    return compare_Div(node,other);
  elif ins(node,ast.FloorDiv):
    return compare_FloorDiv(node,other);
  elif ins(node,ast.BitOr):
    return compare_BitOr(node,other);
  elif ins(node,ast.BitXor):
    return compare_BitXor(node,other);
  elif ins(node,ast.BitAnd):
    return compare_BitAnd(node,other);
  elif ins(node,ast.And):
    return compare_And(node,other);
  elif ins(node,ast.Or):
    return compare_Or(node,other);
  elif ins(node,Impl):
    return compare_Impl(node,other);
  elif ins(node,ast.Not):
    return compare_Not(node,other);
  elif ins(node,ast.USub):
    return compare_USub(node,other);
  elif ins(node,ast.Eq):
    return compare_Eq(node,other);
  # ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
  elif ins(node,ast.NotEq):
    return compare_NotEq(node,other);
  elif ins(node,ast.Lt):
    return compare_Lt(node,other);
  elif ins(node,ast.LtE):
    return compare_LtE(node,other);
  elif ins(node,ast.Gt):
    return compare_Gt(node,other);
  elif ins(node,ast.GtE):
    return compare_GtE(node,other);
  elif ins(node,ast.Is):
    return compare_Is(node,other);
  elif ins(node,ast.IsNot):
    return compare_IsNot(node,other);
  elif ins(node,ast.In):
    return compare_In(node,other);
  elif ins(node,ast.NotIn):
    return compare_NotIn(node,other);
  #elif ins(node,ast.GeneratorExp):
  #  return compare_GeneratorExp(node,other)
  #elif ins(node,ast.ExtSlice):
  #  return compare_ExtSlice(node,other)
  elif ins(node,ast.Invert):
    return compare_Invert(node,other)
  elif ins(node, ast.IfExp):
    return compare_IfExp(node,other)
  #elif ins(node, ast.Slice):
  #  return compare_Slice(node,other)
  else:
    raise ValueError("I don't know what kind of object is this!", ast.dump(node));
