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

def compare_Starred(node,other):
  if ins(other,ast.Starred):
    return compare(node.value,other.value)
  else:
    return False

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


def compare_Comprehension(node, other):
  if compare(node.target, other.target):
    if compare(node.iter, other.iter):
      if compare(node.ifs, other.ifs):
        return True
  return False

def compare_ListComp(node, other):
  if ins(other, ast.ListComp):
    if compare(node.elt, other.elt):
      if len(node.generators) == len(other.generators):
        for i in range(0,len(node.generators)):
          if not(compare_Comprehension(node.generators[i], other.generators[i])):
            return False
          return True
  return False

def compare_SetComp(node, other):
  if ins(other, ast.SetComp):
    if compare(node.elt, other.elt):
      if len(node.generators) == len(other.generators):
        for i in range(0,len(node.generators)):
          if not(compare_Comprehension(node.generators[i], other.generators[i])):
            return False
          return True
  return False

def compare_DictComp(node, other):
  if ins(other, ast.DictComp):
    if compare(node.elt, other.elt):
      if len(node.generators) == len(other.generators):
        for i in range(0,len(node.generators)):
          if not(compare_Comprehension(node.generators[i], other.generators[i])):
            return False
          return True
  return False

# Certain expressions. May occur as Base operands.

def compare_GeneratorExp(node,other):
  if ins(other,ast.GeneratorExp):
    if compare(node.elt,other.elt) and len(node.generators) == len(other.generators):
      for i in range(0,len(node.generators)):
        # TODO: Figure value equality for comprehensions
        #if node.generators[i] != other.generators[i]:
        if not(compare_Comprehension(node.generators[i], other.generators[i])):
          return False
      return True
  return False

# TODO: ExtSlice has been deprecated
# def compare_ExtSlice(node, other):
#  return False 

def compare_IfExp(node, other):
  if ins(other, ast.IfExp):
   return compare(node.test,other.test) and compare(node.body,other.body) and compare(node.orelse,other.orelse)
  else:
   return False

# Put strings to set and compare
# ast.Const: just string
# ast.Name: string of id
# ast.Attribute: string of id + attr
# ast.Call: string of func.id and args
# ast.Subscript: string of value and slice
# ast.Tuple: string of elts
def compare_Set(node, other):
  a = {*()} # empty set
  b = {*()}

  if not ins(other, ast.Set):
    return False

  def getString(x):
    if ins(x, ast.Constant):
      return "Const_" + str(x.value)
    elif ins(x, ast.Name):
      return "Name_" + x.id
    elif ins(x, ast.Attribute):
      return "Attribute_" + getString(x.value) + "_" + x.attr
    elif ins(x, ast.Call):
      r = "Call" + getString(x.func)
      for arg in x.args:
        r += "_" + getString(arg)
      return r
    elif ins(x, ast.Subscript):
      return "Subscript_" + getString(x.value) + "_" + getString(x.slice)
    elif ins(x, ast.Tuple):
      r = "Tuple"
      for e in x.elts:
        r += "_" + getString(e)
      return r

    else:
      print("\n!!x =",ast.dump(x))
      print("!!node =",ast.dump(node))
      print("!!other =",ast.dump(other))
      assert False, "PANIC: an element in ast.Set is not ast.Constant"   

  for elt in node.elts:
    a.add(getString(elt))

  for elt in other.elts:
    b.add(getString(elt))

  #print("a = ", a)
  #print("b = ", b)
  return set(a) == set(b)


# TODO: handle comparison of Slice expressions more intelligently
# def compare_Slice(node, other):
  #if ins(other, ast.Slice):
  #  return compare(node.lower,other.lower) and compare(node.upper,other.upper) and compare(node.step,other.step)
  #else:
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
  if ins(node,ast.Starred):
    return compare_Starred(node,other);
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
  elif ins(node,ast.GeneratorExp):
    return compare_GeneratorExp(node,other)
  #elif ins(node,ast.ExtSlice):
  #  return compare_ExtSlice(node,other)
  elif ins(node,ast.Invert):
    return compare_Invert(node,other)
  elif ins(node, ast.IfExp):
    return compare_IfExp(node,other)
  #elif ins(node, ast.Slice):
  #  return compare_Slice(node,other)
  elif ins(node, ast.ListComp):
    return compare_ListComp(node, other)
  elif ins(node, ast.SetComp):
    return compare_SetComp(node, other)  
  elif ins(node, ast.DictComp):
    return compare_DictComp(node, other)  
  elif ins(node, ast.Set):
    return compare_Set(node, other)
  else:
    if node is None and other is None:
      return True
    if node is None or other is None:
      return False

    if type(node) is list and type(other) is list:
      n = 0
      r = True
      while n < len(node):
        r = r and compare(node[n], other[n])
        n += 1
      return r

    # TODO: ???
    #print(ast.dump(node))
    #print(ast.dump(other))
    if ins(node, ast.Mod) and ins(other, ast.Mod):
      return True
    if ins(node, ast.Pow) and ins(other, ast.Pow):
      return True
    # quick fix for ast.Slice(upper=ast.Constant(value=4))
    if ins(node, ast.Slice) and ins(other, ast.Slice):
      return compare(node.upper,other.upper) and compare(node.lower, other.lower)
      return node.upper.value == other.upper.value
    if ins(node, ast.SetComp):
      if ins(other, ast.SetComp):
        assert False, "Won't reach here. We now deal with SetComp in compare_SetComp()"
        return True
      return False

    if ins(node, ast.DictComp):
      if ins(other, ast.DictComp):
        assert False, "Won't reach here. We now deal with DictComp in compare_DictComp()"
        return True
        assert False, "PANIC: Have to compare 2 ast.DictComp"
      return False
    raise ValueError("I don't know what kind of object is this!", ast.dump(node));
