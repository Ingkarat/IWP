import ast

from equality import Impl

def ins(obj,cls):
  return isinstance(obj,cls);

def parens(arg): 
  return '('+arg+')';

def is_operand(node):
  if ins(node,ast.Name) or ins(node,ast.Attribute) or ins(node,ast.Constant) or ins(node,ast.Call) or ins(node,ast.List) or ins(node,ast.Dict) or ins(node,ast.Tuple) or ins(node,ast.Subscript) or ins(node,ast.Index) or ins(node,ast.IfExp) or ins(node, ast.Set):
    return True;
  else: 
    return False;

def is_operator(node):
  if ins(node,ast.Compare) or ins(node,ast.UnaryOp) or ins(node,ast.BinOp) or ins(node,ast.BoolOp):
    return True;
  else:
    return False;

# TODO. Check this out, precedence and all. Possibly add semantic checks for expressions.

# returns precedence,string tuple;
def pprint_op(node):
  if ins(node, ast.BoolOp): 
    return pprint_boolop(node);
  elif ins(node, ast.BinOp): 
    return pprint_binop(node);
  elif ins(node, ast.UnaryOp):
    return pprint_unaryop(node);
  elif ins(node, ast.Compare):
    return pprint_compop(node);
  elif is_operand(node):
    # it's an operand, var, attribute, call, etc
    return 10, pprint_operand(node);
  else:
    if ins(node, ast.ListComp):
      return 99, "THIS NODE INCLUDE ListComp WHICH IS NOT SUPPORED YET"
    if ins(node, ast.SetComp):
      return 99, "THIS NODE INCLUDE SetComp WHICH IS NOT SUPPORED YET"
    if ins(node, ast.DictComp):
      return 99, "THIS NODE INCLUDE DictComp WHICH IS NOT SUPPORED YET"
    if ins(node, ast.GeneratorExp):
      return 99, "THIS NODE INCLUDE GeneratorExp WHICH IS NOT SUPPORED YET"

    raise ValueError("pprint_op: I don't know what kind of node this is: "+ast.dump(node));

def pprint_binop(node):
  #operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
  # TODO: fill in all operators!
  if ins(node.op,ast.Add): op_str = '+'; prec = 6;
  elif ins(node.op,ast.Sub): op_str ='-'; prec = 6;
  elif ins(node.op,ast.Mult): op_str ='*'; prec = 7;
  elif ins(node.op,ast.Div): op_str ='/'; prec = 7;
  elif ins(node.op,ast.FloorDiv): op_str ='/'; prec = 7;
  elif ins(node.op,ast.BitOr): op_str ='|'; prec = 8; # TODO: check precedence of Bitwise ops!
  elif ins(node.op,ast.BitXor): op_str ='^'; prec = 8;
  elif ins(node.op,ast.BitAnd): op_str ='&'; prec = 8; 
  elif ins(node.op,ast.Mod): op_str ='%'; prec = 7; #TODO: precedence of mod? 
  elif ins(node.op,ast.Pow): op_str = "**"; prec = 7; #TODO: precedence of Pow?
  elif ins(node.op,ast.LShift): op_str = "<<"; prec = 9 # TODO: precedence of LShift?
  elif ins(node.op,ast.RShift): op_str = ">>"; prec = 9 # TODO: precedence of RShift?
  else: 
    raise ValueError("TODO: unrecognized binary operator: ",ast.dump(node.op));  
  prec_left, str_left = pprint_op(node.left); # Operand type: BinOp or basic_node 
  prec_right, str_right = pprint_op(node.right); # Operand type: BinOp or basic_node
  if prec_left < prec: 
    str_left = parens(str_left);
  if prec_right <= prec: 
    str_right = parens(str_right);
  return prec, str_left+op_str+str_right;  

def pprint_boolop(node):
  #boolop = And | Or #All values should be cmpoperators!!!
  if ins(node.op, ast.And): op_str = ' and '; prec = 1;
  elif ins(node.op, ast.Or): op_str = ' or '; prec = 0;
  elif ins(node.op, Impl): op_str = '  =>  '; prec = 0;
  else: 
      raise ValueError("Unrecognized boolean op");
  result = ''
  for value in node.values:
    prec_v, str_v = pprint_op(value); #  Not or Compare; Cannot be a basic
    if prec_v <= prec:
      str_v = '('+str_v+')'
    result+=str_v;
    if value != node.values[-1]: # it's not the last 
      result+=op_str;
  return prec, result;

def pprint_unaryop(node):
  #unaryop = Invert | Not | UAdd | USub
  #TODO: all the operators
  if ins(node.op,ast.Not): op_str = 'NOT'; prec = 10;
  elif ins(node.op,ast.USub): op_str = '-'; prec = 10;
  elif ins(node.op,ast.Invert): op_str = "~"; prec = 10;
  else: 
    raise ValueError("TODO: Unrecognized unary op", node.op);
  prec_operand, str_operand = pprint_op(node.operand); # Compare; Can't be a basic or BinOp
  result = op_str+'('+str_operand+')'; 
  return prec, result;

def get_str_compop(node, index):
  if 0:
    if ins(node.ops[0],ast.Eq): op_str = ' == ';
    elif ins(node.ops[0],ast.NotEq): op_str =' =/= ';
    elif ins(node.ops[0],ast.Lt): op_str =' < ';
    elif ins(node.ops[0],ast.LtE): op_str =' <= ';
    elif ins(node.ops[0],ast.Gt): op_str =' > ';
    elif ins(node.ops[0],ast.GtE): op_str =' >= ';
    elif ins(node.ops[0],ast.Is): op_str =' Is ';
    elif ins(node.ops[0],ast.IsNot): op_str =' IsNot ';
    elif ins(node.ops[0],ast.In): op_str =' In ';
    elif ins(node.ops[0],ast.NotIn): op_str =' NotIn ';
    else:
      raise ValueError("Unrecognized comparison operator: "+node,op);
    return op_str;
  else:
    if ins(node.ops[index],ast.Eq): op_str = ' == ';
    elif ins(node.ops[index],ast.NotEq): op_str =' =/= ';
    elif ins(node.ops[index],ast.Lt): op_str =' < ';
    elif ins(node.ops[index],ast.LtE): op_str =' <= ';
    elif ins(node.ops[index],ast.Gt): op_str =' > ';
    elif ins(node.ops[index],ast.GtE): op_str =' >= ';
    elif ins(node.ops[index],ast.Is): op_str =' Is ';
    elif ins(node.ops[index],ast.IsNot): op_str =' IsNot ';
    elif ins(node.ops[index],ast.In): op_str =' In ';
    elif ins(node.ops[index],ast.NotIn): op_str =' NotIn ';
    else:
      raise ValueError("Unrecognized comparison operator: "+node,op);
    return op_str;

def pprint_compop(node):
  #cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
  #expr left, cmpop* ops, expr* comparators 
  result = ''; prec = 5;
  prec_left, str_left = pprint_op(node.left); # This is basic, or BinOp 
  assert prec_left > prec;
  result+= str_left;
  i = 0;
  for op in node.ops:
    str_op = get_str_compop(node,i);  
    prec_right, str_right = pprint_op(node.comparators[i]); # This basic or BinOp
    assert prec_right > prec;
    result+= str_op+str_right;
    i+=1;
  return prec, result  

def pprint_operand(node):
  if ins(node,ast.Name):
    return node.id;
  elif ins(node,ast.Attribute):
    value = pprint_top(node.value);
    return value+'.'+node.attr;
  elif ins(node,ast.Constant):
    if ins(node.value,str): return "\'"+node.value+"\'";
    return str(node.value);
  # Composite basic nodes...
  elif ins(node,ast.Call):
    # TODO: Printed args; need to print keywords (i.e., named arguments) too.
    func = pprint_top(node.func);
    args = '';
    for arg in node.args:
      args += pprint_top(arg);
      if node.args[-1]!=arg:
        args += ',';
    #for kw in node.keywords:

    return func+'('+args+')';
  elif ins(node,ast.List) or ins(node,ast.Tuple):
    elts = '';
    for elem in node.elts:
      elts += pprint_top(elem);
      if node.elts[-1]!=elem:
        elts += ',';
    return '['+elts+']';
  elif ins(node,ast.Dict):
    elts = ''; c = 0;
    for key in node.keys:
      elts += pprint_top(key)+':'+pprint_top(node.values[c]);
      if node.keys[-1]!=key: elts +=',';
      c+=1;
    return '{'+elts+'}'; 
  elif ins(node,ast.Subscript): 
    val = pprint_top(node.value);
    slice = pprint_top(node.slice);
    return val+'['+slice+']'
  elif ins(node,ast.Index):
    return pprint_top(node.value);
  elif ins(node,ast.IfExp):
    return "IfExp("+pprint_top(node.test)+","+pprint_top(node.body)+","+pprint_top(node.orelse)+")";
  elif ins(node, ast.Set):
    return pprint_set(node)  
  else: 
    raise ValueError("pprint_operand: I don't know what Im doing: ",ast.dump(node));

def pprint_set(node):
  ret = "{"
  for elt in node.elts:
    ret += pprint_top(elt)
    ret += ", "
  return ret[:-2] + "}"

def pprint_top(node):
  if is_operand(node):
    return pprint_operand(node);
  elif is_operator(node): 
    p, str = pprint_op(node);
    return str;
  else:
    return "";
    #raise ValueError("unionwn node",node);
    #expr left, cmpop* ops, expr* comparators
    
def pprint_list(formulas):
  result = '';
  for formula in formulas:
    result+='-- '+pprint_top(formula);
    if formula != formulas[-1]: 
      result+='\n and\n';
  result+='\n';
  return result;
