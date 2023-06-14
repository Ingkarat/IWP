import ast
import copy
import equality as eq
import hacks
import warnings

from hacks import TRUE
from hacks import FALSE

def ins(obj,cls):
  return isinstance(obj,cls);

def is_operator(node):
  if ins(node,ast.Compare) or ins(node,ast.UnaryOp) or ins(node,ast.BinOp) or ins(node,ast.BoolOp):
    return True;
  else:
    return False;

def simplify_op(node):
  #print("\n[ARRRRRRRRRRR]simplify_op: ",ast.dump(node));
  if ins(node, ast.BoolOp): 
    return simplify_boolop(node);
  elif ins(node, ast.BinOp): 
    return simplify_binop(node);
  elif ins(node, ast.UnaryOp):
    return simplify_unaryop(node);
  elif ins(node, ast.Compare):
    return simplify_compop(node);
  else:
    raise ValueError("I don't know what kind of node this is: ",node);

def simplify_binop(node):
  #operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
  # TODO: fill in all operators!
  if ins(node.op,ast.Add): 
    if eq.compare(node.left,ast.Constant(value=0)): 
      return node.right;
    elif eq.compare(node.right,ast.Constant(value=0)): 
      #print("\n\nHere")
      #print(ast.dump(node))
      return node.left;
  elif ins(node.op,ast.Sub): 
    if eq.compare(node.right,ast.Constant(value=0)): 
      return node.left;    
    elif eq.compare(node.left,node.right):
      return ast.Constant(value=0);
  elif ins(node.op,ast.Mult): 
    if eq.compare(node.left,ast.Constant(value=0)): 
      return node.left;
    elif eq.compare(node.right,ast.Constant(value=0)): 
      return node.right;
    elif eq.compare(node.left,ast.Constant(value=1)): 
      return node.right;
    elif eq.compare(node.right,ast.Constant(value=1)): 
      return node.left;
  elif ins(node.op,ast.Div) or ins(node.op,ast.FloorDiv): 
    if eq.compare(node.left,ast.Constant(value=0)): 
      return node.left;
    elif eq.compare(node.right,ast.Constant(value=0)): 
      raise ValueError("DIVISION BY ZERO! ", ast.dump(node));
  elif ins(node.op,ast.BitOr) or ins(node.op,ast.BitXor) or ins(node.op,ast.BitAnd):
    #TODO: Explore simplification of Bitwise ops?
    return node;
  elif ins(node.op, ast.Mod):
    return node
  elif ins(node.op, ast.Pow):
    return node
  else:
    #print(ast.dump(node),ast.dump(op)); 
    raise ValueError("TODO: unrecognized binary operator: ", ast.dump(node));  
  return node;

def simplify_boolop(node):
  #boolop = And | Or #All values should be cmpoperators!!!
  #new_node = copy.deepcopy(node);
  new_node = copy.copy(node);
  if ins(node.op, ast.And): 
    new_node.values = [];
    for value in node.values:
      if eq.compare(value,FALSE): 
        return FALSE;
      elif eq.compare(value,TRUE):
        pass;
      else:
        new_node.values.append(value);
    if new_node.values == []:
      return TRUE;
    elif len(new_node.values) == 1:
      return new_node.values[0];
    else:
      return new_node;
  elif ins(node.op, ast.Or):
    new_node.values = [];
    for value in node.values:
      if eq.compare(value,TRUE): 
        return TRUE;
      elif eq.compare(value,FALSE): 
        pass;
      else:
        new_node.values.append(value);
    if new_node.values == []:
      return FALSE;
    elif len(new_node.values) == 1:
      return new_node.values[0];
    else:
      return new_node;
  elif ins(node.op, eq.Impl):
    ante = node.values[0];
    cons = node.values[1];
    # For ImpL, if either side is Tuple, we can safely assign that to TRUE (right?)
    if ins(ante, ast.Tuple):
      ante = TRUE
    if ins(cons, ast.Tuple):
      cons = TRUE

    if eq.compare(ante,FALSE) or eq.compare(cons,TRUE):
      return TRUE;
    elif eq.compare(ante,TRUE):
      return cons;
    elif eq.compare(cons,FALSE):
      return ast.UnaryOp(op=ast.Not(),operand=ante);
    else:
      return new_node;
  else: 
    raise ValueError("Unrecognized boolean op");  

def simplify_unaryop(node):
  #unaryop = Invert | Not | UAdd | USub
  if ins(node.op, ast.Not): 
    if eq.compare(node.operand,TRUE): 
      return FALSE;
    elif eq.compare(node.operand,FALSE): 
      return TRUE;
    elif ins(node.operand,ast.UnaryOp) and ins(node.operand.op,ast.Not):
      aa = node.operand.operand
      #print("5555555555: \n", ast.dump(aa))
      #return simplify_op(node.operand.operand); # NOT(NOT(A)) = A
      return node.operand.operand; # NOT(NOT(A)) = A

  return node;
 
def simplify_compop(node):
  #print("\n[ARRRRRRRRRRR]simplify_compop: ",ast.dump(node));
  #cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
  #expr left, cmpop* ops, expr* comparators 
  if len(node.ops) != 1 or len(node.comparators) != 1:
    return node; # we can only simplify x op y but not x op1 y1 op2 y2 etc.
 
  op = node.ops[0];
  left = node.left;
  right = node.comparators[0];
  if ins(op,ast.Eq): 
    if eq.compare(left,right): # if they are equal
      return TRUE;
  elif ins(op,ast.NotEq): 
    if eq.compare(left,right): # if they are equal
      return FALSE;
  elif ins(op,ast.Lt): 
    pass
  elif ins(op,ast.LtE): 
    pass
  elif ins(op,ast.Gt): 
    pass
  elif ins(op,ast.GtE): 
    pass
  elif ins(op,ast.Is): #TODO: Check this code.
    if (ins(left,ast.List) or ins(left,ast.Dict) or ins(left,ast.Tuple)) and ins(right,ast.Constant):
      return FALSE;
    elif (ins(right,ast.List) or ins(right,ast.Dict) or ins(right,ast.Tuple)) and ins(left,ast.Constant):
      return FALSE;
  elif ins(op,ast.IsNot): 
    if (ins(left,ast.List) or ins(left,ast.Dict) or ins(left,ast.Tuple)) and ins(right,ast.Constant):
      return TRUE;
    elif (ins(right,ast.List) or ins(right,ast.Dict) or ins(right,ast.Tuple)) and ins(left,ast.Constant):
      return TRUE;
  elif ins(op,ast.In):
    if ins(right,ast.List) or ins(right,ast.Tuple):
     for elem in right.elts:
       if eq.compare(left,elem): 
         return TRUE;
  elif ins(op,ast.NotIn): #TODO: This is Not fully sound. If left is a side-effecting expression... 
    if ins(right,ast.List) or ins(right,ast.Tuple):
     for elem in right.elts:
       if eq.compare(left,elem):
         return FALSE;
  else:
    raise ValueError("Unrecognized comparison operator: "+node,op);  
  return node;

def simplify(node):

  # f_simplify (always do simplify)
  # XXXreturn nodeXXX
  #return node

  if 0:
    if is_operator(node): 
      return simplify_op(node);
    else:
      return hacks.hack(node);

  if True:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      interpreted_node = interpret_node_natively(node)
  else:
    interpreted_node = node

  if is_operator(interpreted_node): 
    return simplify_op(interpreted_node);
  else:
    # TODO: hacks should be obsolete with interpretation
    return hacks.hack(interpreted_node);

# ==== programatic simplification =====

def interpret_node_natively(node):
  # will return simplified node... and if boolean, will return TRUE or FALSE
  # print("Before interpret: ", ast.dump(node))

  # f_simplify (return if we DONT do CE)
  # return node

  try:
    code = ast.unparse(node)

    # Sometime we can (or have to?) resolve dict {}.get() even though we cant resolve some elements in the dict 
    # ex. {'accept_sparse':['csr','csc'],'ensure_2d':True,'allow_nd':True,'dtype':None,'y_numeric':is_regressor(self)}.get('accept_sparse',False),str)
    # Have to detect here i guess
    # Ugly code 
    good = False
    if ins(node, ast.Call):
      if ins(node.func, ast.Attribute):
        if ins(node.func.value, ast.Dict):
          if node.func.attr == "get":
            if "}.get(" in code:
              good = True
              if ins(node.args[0], ast.Name): # For kwarg, key can only be ast.Constant
                good = False
    if good:
      #print("\n> IN simplify.py, dictionary case for .get()")
      #print(ast.dump(node))
      #print(ast.unparse(node))
      result = node.args[1]
      num = 0
      keyName = ""
      if ins(node.args[0], ast.Constant):
        keyName = node.args[0].value
      elif ins(node.args[0], ast.Name):
        keyName = node.args[0].id
        assert False, "Won't reach here"
      else:
        assert False, "What else? in interpret_node_natively"
      #print(">> ", ast.dump(node))
      #print("<< ", ast.unparse(node))
      #print("!! ", keyName)
      for l in node.func.value.keys:
        #print("> ", ast.dump(l), ast.dump(node.args[0]))
        if l is None:
          if keyName is None:
            result = node.func.value.values[num]
        elif l.value == keyName:
          result = node.func.value.values[num]
          #print("FOUND:", ast.dump(result))
        num += 1
      #print("    Got: ", ast.dump(result))
      return result      


  except KeyError:    
    # print("Aha, error likely due to Impl: ")
    pass
  else:
    # print("The source segment: ",code);
    try:
      #print("Code =", code)
      if code != "None.kind":
        exec('import numpy as np; global val; val = '+code)
      else:
        print("?????????????????? in simplify.py")
    except AttributeError:
      pass
    except IndexError:
      pass
    except NameError:
      # print("Cannot resolve name")
      pass
    except SyntaxError:
      # print("Syntax error")
      pass
    except TypeError:
      # print("Type error")
      pass
    except:
      pass
    else:
      # print("EVALUATED! The val is: ",val);
      try:
        # probably safe to assume that if it reaches this, it will be just "{}.get()" that can be eval.
        if "}.get(" in code:
          #print("\n> IN simplify.py, dictionary case for .get()")
          #print(ast.dump(node))
          #print("   ", ast.unparse(node))
          good = False
          if ins(node, ast.Call):
            if ins(node.func.value, ast.Dict) and node.func.attr == "get":
              good = True
          assert good, "node is not ast.Call with ast.Dict and .get (in simplify.py)"
          result = node.args[1]
          #print("Default is", ast.dump(result))
          num = 0
          for l in node.func.value.keys:
            if l.value == node.args[0].value:
              result = node.func.value.values[num]
              #print("FOUND:", ast.dump(result))
            num += 1
          #print("    Got: ", ast.dump(result))
          return result


        # Successfully evaluated a node! Return the node when constant! 
        #print("\nEVALUATED ", code," into val =",val,ast.dump(ast.parse(str(val)).body[0].value))
        parse_back = ast.parse(str(val))
        #print("parse_back:", ast.dump(parse_back), ast.unparse(parse_back))
        #print("code:", code)
        #print("node:", ast.dump(node))
        if ins(node,ast.Constant):
          return node
        result = parse_back.body[0].value
        #print(ast.dump(result))
        #print("\nEVALUATED ", code," into val =",ast.dump(result))
        if len(parse_back.body) == 1 and ins(result,ast.Constant): # for dictionary, it MIGHT NOT be an ast.Constant
          return result 
      except SyntaxError:
        pass

  return node


# ==== utils =====

def negate(test):
  result = ast.UnaryOp(op=ast.Not(),operand=test);
  return simplify(result);

def cons_and(op1,op2):
  result = ast.BoolOp(op=ast.And(),values=[op1,op2]);
  return simplify(result);

def cons_impl(ante,cons):
  result = ast.BoolOp(op=eq.Impl(),values=[ante,cons]);
  return simplify(result);

def ins_and(operand):
  if ins(operand,ast.BoolOp) and ins(operand.op,ast.And):
    return True;
  else:
    return False;

# factors out test => A^B ^ !test => C^B <=> (test=>A ^ !test=>C) ^ B
def factor_out(test, wp1, wp2):
  #print(" Factoring out ",ast.dump(wp1),'\n AND \n', ast.dump(wp2));   
  wp1_left,wp2_left,common = factor_all(wp1,wp2); # factor_tail(wp1,wp2);
  true_branch = cons_impl(test,wp1_left);
  false_branch = cons_impl(negate(test),wp2_left);
  return cons_and(cons_and(true_branch,false_branch),common);

def is_in(e, common):
  for elem in common:
    if eq.compare(e,elem):
      return True;
  return False;

# constructs a right-recursive "and" object
# lis: [c1,c2,c3] and c1 and (c2 and c3) if c1,c2,c3 do not occur in common
def cons_and_from_list(lis,common):
  result = TRUE; l = len(lis);
  for i in range(0,l):
    if not(is_in(lis[l-1-i],common)):
      result = cons_and(lis[l-1-i],result);
  return result;

def flatten_and(cond):
  if not(ins_and(cond)): 
    return [cond];
  else:
    return flatten_and(cond.values[0])+flatten_and(cond.values[1]);

def factor_all(wp1, wp2):
  lis1 = flatten_and(wp1);
  lis2 = flatten_and(wp2);
  common = [];
  for e1 in lis1:
    for e2 in lis2:
      if eq.compare(e1,e2) and not(is_in(e1,common)):
        common.append(e1);
  # print("COMMON: ", common);
  wp1_new = cons_and_from_list(lis1,common); 
  wp2_new = cons_and_from_list(lis2,common);
  # print("HERE", ast.dump(wp1_new), ' and ', ast.dump(wp2_new));
  com = cons_and_from_list(common,[]);
  return wp1_new,wp2_new,com;
 

