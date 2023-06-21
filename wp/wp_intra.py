#!/usr/local/bin/python3.9

import ast
import sys
import os
import shutil
import copy
import pprint2

import simplify as si
import equality
import hacks
import raise_analysis
import helper
import config

import REF_ref_immutability

from simplify import FALSE
from simplify import TRUE
from simplify import simplify
from equality import Impl

def ins(obj,cls):
  return isinstance(obj,cls);

def occurs_in_formula(wp,target):
  if ins(wp,ast.Name): 
    return equality.compare(wp,target)
  elif ins(wp,ast.Attribute) and ins(wp.value,ast.Name) and wp.value.id == 'self':
    return equality.compare(wp,target)
  elif ins(wp,ast.Attribute) and ins(wp.value,ast.Name) and wp.value.id == 'sparse': #check sparse.issparse()
    return equality.compare(wp,target)
  if ins(wp,ast.Subscript): #probably not the most precise way
    return equality.compare(wp,target)
  else:
    if wp is None:
      return False
    for attr in wp.__dict__.keys():
      if ins(wp.__dict__[attr],ast.AST):
        if occurs_in_formula(wp.__dict__[attr],target):
          return True
      elif ins(wp.__dict__[attr],list):
        for elt in wp.__dict__[attr]:
           if occurs_in_formula(elt,target): 
             return True
    return False

# requires: target is either a name or an attribute of self.
# replaces every occurrence of target with expr in wp
# ll is list of possible kwarg. we dont replace their subscript here. Use KWARG_replace_in_formula isntead
def replace_in_formula_REC(wp,target,expr):
  # This is a HUGE UGLY HACK but hey it's Python.
  # print("replace_in_formula, replacing ", ast.dump(target), "in", ast.dump(wp), "with", ast.dump(expr));
  if ins(wp, ast.Subscript):
    #if ins(wp.value, ast.Name):
    #  if wp.value.id in ll:
    #    return copy.deepcopy(wp)
    if equality.compare(wp,target):
      return copy.deepcopy(expr)
    else:
      return copy.deepcopy(wp)

  if ins(wp,ast.Name): # found a variable!
    if equality.compare(wp,target): # ins(target,ast.Name) and wp.id == target.id:
      return copy.deepcopy(expr)
    else:
      return copy.deepcopy(wp);
  elif ins(wp,ast.Attribute) and ins(wp.value,ast.Name) and wp.value.id == 'self': # searching for a ref to self.attr
    if equality.compare(wp,target):
      # print("I am HERE, replacing :", ast.dump(wp), "with", ast.dump(target));
      return copy.deepcopy(expr) # need a deep copy
    else:
      return copy.deepcopy(wp) 
  else:
    #wp_replace = copy.deepcopy(wp);
    wp_replace = copy.copy(wp)
    if wp is None:
      return wp_replace
    for attr in wp.__dict__.keys():
      if ins(wp.__dict__[attr],ast.AST):
        attr_replace = replace_in_formula_REC(wp.__dict__[attr],target,expr)
        wp_replace.__dict__[attr] = attr_replace
      elif ins(wp.__dict__[attr],list):
        new_list = [];
        for elt in wp.__dict__[attr]:
           elt_replace = replace_in_formula_REC(elt,target,expr)
           new_list.append(elt_replace)
        wp_replace.__dict__[attr] = new_list
    wp_replace = simplify(wp_replace)
    return wp_replace

# Wrapper to set the limit
def replace_in_formula(wp,target,expr):
  wp_new = replace_in_formula_REC(wp,target,expr)
  # f_prune
  #if 1:
  if 1:
    impl_limit = config.IMPL_LIMIT   # default = 200
    msg = "FILTERED WP. IT HAS > " + str(impl_limit) + " IMPLICATIONS AT SOME POINT."
    if helper.impl_counter(wp_new) >= impl_limit:
      wp_new = ast.Constant(value = msg)
  return wp_new


# Specific case for dealing with Dict() in svm/_base.py:_get_liblinear_solver_type()
# Ugliest hack i've ever coded in my life
d_Target = ast.BoolOp(op=ast.And(), values=[ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=ast.Call(func=ast.Attribute(value=ast.Dict(keys=[ast.Constant(value='logistic_regression'), ast.Constant(value='hinge'), ast.Constant(value='squared_hinge'), ast.Constant(value='epsilon_insensitive'), ast.Constant(value='squared_epsilon_insensitive'), ast.Constant(value='crammer_singer')], values=[ast.Dict(keys=[ast.Constant(value='l1'), ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False)], values=[ast.Constant(value=6)]), ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=0), ast.Constant(value=7)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=True)], values=[ast.Constant(value=3)])]), ast.Dict(keys=[ast.Constant(value='l1'), ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False)], values=[ast.Constant(value=5)]), ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=2), ast.Constant(value=1)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=True)], values=[ast.Constant(value=13)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=11), ast.Constant(value=12)])]), ast.Constant(value=4)]), attr='get', ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ast.Constant(value=None)], keywords=[]), ops=[ast.Is()], comparators=[ast.Constant(value=None)])), ast.BoolOp(op=ast.And(), values=[ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Attribute(value=ast.Dict(keys=[ast.Constant(value='logistic_regression'), ast.Constant(value='hinge'), ast.Constant(value='squared_hinge'), ast.Constant(value='epsilon_insensitive'), ast.Constant(value='squared_epsilon_insensitive'), ast.Constant(value='crammer_singer')], values=[ast.Dict(keys=[ast.Constant(value='l1'), ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False)], values=[ast.Constant(value=6)]), ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=0), ast.Constant(value=7)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=True)], values=[ast.Constant(value=3)])]), ast.Dict(keys=[ast.Constant(value='l1'), ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False)], values=[ast.Constant(value=5)]), ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=2), ast.Constant(value=1)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=True)], values=[ast.Constant(value=13)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=11), ast.Constant(value=12)])]), ast.Constant(value=4)]), attr='get', ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ast.Constant(value=None)], keywords=[]), attr='get', ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ast.Constant(value=None)], keywords=[]), ops=[ast.Is()], comparators=[ast.Constant(value=None)])), ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Attribute(value=ast.Dict(keys=[ast.Constant(value='logistic_regression'), ast.Constant(value='hinge'), ast.Constant(value='squared_hinge'), ast.Constant(value='epsilon_insensitive'), ast.Constant(value='squared_epsilon_insensitive'), ast.Constant(value='crammer_singer')], values=[ast.Dict(keys=[ast.Constant(value='l1'), ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False)], values=[ast.Constant(value=6)]), ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=0), ast.Constant(value=7)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=True)], values=[ast.Constant(value=3)])]), ast.Dict(keys=[ast.Constant(value='l1'), ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False)], values=[ast.Constant(value=5)]), ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=2), ast.Constant(value=1)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=True)], values=[ast.Constant(value=13)])]), ast.Dict(keys=[ast.Constant(value='l2')], values=[ast.Dict(keys=[ast.Constant(value=False), ast.Constant(value=True)], values=[ast.Constant(value=11), ast.Constant(value=12)])]), ast.Constant(value=4)]), attr='get', ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ast.Constant(value=None)], keywords=[]), attr='get', ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ast.Constant(value=None)], keywords=[]), attr='get', ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ast.Constant(value=None)], keywords=[]), ops=[ast.Is()], comparators=[ast.Constant(value=None)]))])])
d_Replacement = ast.BoolOp(op=ast.Or(), values=[ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='logistic_regression')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l1')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=False)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='logistic_regression')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='hinge')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=True)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_hinge')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l1')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=False)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_hinge')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='epsilon_insensitive')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=True)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_epsilon_insensitive')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='multi_class', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='crammer_singer')])])
def replace_dict(wp):
  if ins(wp, ast.Load):
    return copy.deepcopy(wp)
  #print("WP:",ast.dump(wp))
  if equality.compare(wp,d_Target):
    return copy.deepcopy(d_Replacement)
  else:
    wp_replace = copy.copy(wp)
    for attr in wp.__dict__.keys():
      if ins(wp.__dict__[attr],ast.AST):
        attr_replace = replace_dict(wp.__dict__[attr])
        wp_replace.__dict__[attr] = attr_replace;
      elif ins(wp.__dict__[attr],list):
        new_list = []
        for elt in wp.__dict__[attr]:
          elt_replace = replace_dict(elt)
          new_list.append(elt_replace)
        wp_replace.__dict__[attr] = new_list
    wp_replace = simplify(wp_replace)
    return wp_replace


# TOPPPPPP
class Analyzer(ast.NodeVisitor):
    def __init__(self,func_name,wps,function_map, kwarg_defaults, helper_function
        , REF_call_graph_analyzer, REF_function_map, immutability_analyzer, callee_REF_soundness_flag):
      # function under analysis: fully qualified name: file:[Class|None]:name
      self.func_name = func_name
      # to reason about unsoundness
      self.unsound_stack = []
      # to track individual exceptions (either Raise or Call node)
      self.curr_ast_node = None
      self.curr_ast_node_key = None
      # wp stack, holds wp formulas        
      self.stack = []
      # a map from raise_node or call_node to wp for function func_name
      self.wps = wps
      # Callee's declared function args
      #self.callee_F_arg = callee_F_arg

      # ugly but will leave this for now
      self.function_map = function_map
      # String of name of kwarg parameter (**kwargs)? None if not exist.
      self.kwargName = None
      # Default values for kwarg. Map of wp's key to [kwarg's name, Dict of def val]  eg. {[x1,x2,y1,y2,y3]: ["check_params", {...}]}
      # wp's key explained below
      self.kwargDefault = kwarg_defaults

      self.params = []

      # Function that return some of unmodified parameters. Eg. _check_solver
      self.helper_function = helper_function


      # For ref immutability (also the function map might be redundant?)
      self.REF_call_graph_analyzer = REF_call_graph_analyzer
      self.REF_function_map = REF_function_map
      self.immutability_analyzer = immutability_analyzer

      # a map from raise_node to soundness flag (T/F)
      self.REF_soundness_flag = callee_REF_soundness_flag

      # A soundness flag for the current visit of the function AST. Used while working on the function AST node for the CURRENT exception. 
      # Reset each time at the beginning of _FunctionDef_Helper.
      # Become True at the exception node we are working on.
      # This is similar to self.stack that tracks the current WP. In fact self.stack only track 1 exception at a time (never has its len() > 1)
      self.curr_SN_flag = True

    def get_mod_set(self, stmt):

      # case when we skip the ref analysis (set in wp_inter_xxx) to speed up the run
      # TODO: should be removed later
      if not self.REF_call_graph_analyzer:
        return {}

      fragment_immutability_analyzer = REF_ref_immutability.FragmentRefImmutabilityAnalyzer(
        self.REF_call_graph_analyzer.call_graph,
        self.REF_call_graph_analyzer.unresolved,
        self.REF_call_graph_analyzer.libcalls,
        self.REF_function_map,
        self.immutability_analyzer.node_to_type,
        self.immutability_analyzer.self_locs,
        stmt)
      fragment_immutability_analyzer.curr_func = self.func_name
      fragment_immutability_analyzer.collect_constraints()
      fragment_immutability_analyzer.solve_constraints()
      mod_set = fragment_immutability_analyzer.get_mod_set()

      return mod_set

    def get_read_set(self, Qpre):
      
      # case when we skip the ref analysis (set in wp_inter_xxx) to speed up the run
      # TODO: should be removed later
      if not self.REF_call_graph_analyzer:
        return {}

      #print("\n\nRead analyzer for ",self.func_name," and, ",pprint2.pprint_top(Qpre))
      read_analyzer = REF_ref_immutability.ReadAnalyzer(
        Qpre,
        self.func_name,
        self.REF_call_graph_analyzer.libcalls)
      read_analyzer.collect_read_set()
      return read_analyzer.result

    # Naming convention for key of wps
    # [x1,x2,y1,y2,y3...,y,y',y"]: wp
    # x1 = raise node | x2 = func name of that raise node
    # y1 = call node | y2 = func name of that caller node | y3 = func name of calle

    def visit_FunctionDef(self, node):

      #print(ast.dump(node))
      if config.PRINT_SOME_INFO:
        print("\n\nAT visit_FunctionDef:")
        print(node.name)
        print(self.func_name)

      fname = ""
      if ";" in self.func_name: # for lgbm where f-in-f is separatedby ;
        fname = self.func_name.split(';')[-1]
      else:
        fname = self.func_name.split(':')[-1]   
      
      #assert node.name == self.func_name.split(':')[-1] # will fail for case of function def inside function 
      # Do we ignore the f-within-f here because we will properly reach here later? 
      if node.name != fname:
        if fname == "":
          assert False, "fname is not defined. BUG?"
        return

      raise_analyzer = raise_analysis.Raise_Analyzer()
      raise_analyzer.visit(node);
      for raise_node in raise_analyzer.raise_nodes:
        naming = [raise_node, self.func_name]
        if 0:
          if node.name == self.func_name.split(':')[-1]:
            naming = [raise_node, self.func_name]
          else:
            naming = [raise_node, self.func_name+":"+node.name]
        #self.wps[raise_node] = FALSE
        # (Q,S) is (False, True)
        self.wps[tuple(naming)] = FALSE
        self.REF_soundness_flag[tuple(naming)] = True

      self.params = node.args;

      for key in self.wps.keys():
        if len(key) == 2:
          if config.PRINT_DEBUG:
            print("..")
            print(key)
          curr_node = key[0]
          if config.PRINT_DEBUG: print("\n\nEXCEPTION:", ast.dump(curr_node))
        else:
          curr_node = key[-3]
          if config.PRINT_DEBUG: 
            print("\n\nCALL:", ast.dump(curr_node))
            print("!!! key:", key)

        self.curr_ast_node = curr_node
        self.curr_ast_node_key = key
        self._FunctionDef_Helper(node, key[0])

      if 0:
        for curr_node in self.wps.keys(): # This will have call_node too
          if config.PRINT_DEBUG: print("\n\nEXCEPTION:", ast.dump(curr_node))
          self.curr_ast_node = curr_node;
          self._FunctionDef_Helper(node);

    def _FunctionDef_Helper(self, node, exact_exception):
      #[name, args, body, decorator_list, returns?, type_comment?] 
      #if not(node.name == 'fit'): return;
      #print("\nFunctionDef: ",ast.dump(node)); 
      #print("!!!! _FunctionDef_Helper ", ast.dump(node))
      self.unsound_stack.append(False);
      self.stack.append(TRUE);
      assert len(self.stack) == 1
      self.curr_SN_flag = True # Set to True (?) at the start(bottom-up) of the function node

      self.Body_Helper(node.body);
      
      unsound = self.unsound_stack.pop();
      wp = self.stack.pop();
      #print("MAIN: wp = ", ast.dump(wp))

      #if (not equality.compare(wp,TRUE)):
      if True: # I think we still need to keep track of wps that are True too.
        # replaced initial wp in map with wp at top of function
        # self.wps[self.curr_ast_node] = wp
        if "/sklearn/svm/_base.py:None:_get_liblinear_solver_type" in self.func_name:
          haveDict = False
          for f in ast.walk(wp):
            if ins(f,ast.Dict):
              haveDict = True
              break
          if haveDict:
            #wp = ast.BoolOp(op=Impl(), values=[ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=ast.Name(id='multi_class', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='crammer_singer')])), ast.BoolOp(op=ast.Or(), values=[ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='logistic_regression')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l1')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=False)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='logistic_regression')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='hinge')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=True)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_hinge')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l1')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=False)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_hinge')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='epsilon_insensitive')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=True)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_epsilon_insensitive')]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.Compare(left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='crammer_singer')])])])
            wp = ast.BoolOp(op=ast.Or(), values=[ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='logistic_regression')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l1')]), ast.Compare(left=ast.Name(id='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=False)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='logistic_regression')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='hinge')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')]), ast.Compare(left=ast.Name(id='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=True)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_hinge')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l1')]), ast.Compare(left=ast.Name(id='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=False)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_hinge')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='epsilon_insensitive')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')]), ast.Compare(left=ast.Name(id='dual', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=True)])]), ast.BoolOp(op=ast.And(), values=[ast.Compare(left=ast.Name(id='loss', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='squared_epsilon_insensitive')]), ast.Compare(left=ast.Name(id='penalty', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='l2')])]), ast.Compare(left=ast.Name(id='multi_class', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value='crammer_singer')])])

        #simp_wp = simplify(wp)
        self.wps[self.curr_ast_node_key] = wp
        self.REF_soundness_flag[self.curr_ast_node_key] = self.curr_SN_flag

        if config.PRINT_SOME_INFO:
          print('\nThe END of Function ',self.func_name,' and exception/call ',ast.dump(self.curr_ast_node),'\n')
          if ins(self.curr_ast_node, ast.Call):
            print("Exception =", ast.dump(exact_exception))
          print("Soundness Flag =", self.curr_SN_flag)
          # print('\nWP (AST dump):', ast.dump(wp),"\n");
          if config.PRINT_DEBUG: print('WP:', pprint2.pprint_top(wp));
          #print(ast.dump(wp))
          if unsound: print('WARNING: Weakest preconditions may be UNSOUND!');
      #else:
      #  print("MAIN: WP = TRUEEEEEE");

    def isThisValueACurrentAstCall(self, node):
      if not ins(node.value, ast.Call):
        return False
      if not node.value == self.curr_ast_node:
        return False
      return True

    def visit_AugAssign(self, node):
        # This is target += value 
        # TODO: Needs testing!
        # print("\nAugAssign: ",ast.dump(node, include_attributes=True)); 
        self.visit_Assign(ast.Assign(targets=[node.target],value=ast.BinOp(left=node.target,right=node.value,op=node.op)));

    def visit_AnnAssign(self, node):
        # print("\nAnnAssign: ",ast.dump(node, include_attributes=True)); 
        if node.value != None:
          self.visit_Assign(ast.Assign(targets=[node.target],value=node.value));

    def visit_Assign(self, node):
        # targets, value; can have tuple, e.g., x,y = f() and multiple targets x = y = f()  
        # print("\nAssign: ",ast.dump(node));
        # self.visit(node.value);
        super(Analyzer, self).generic_visit(node);

        if not self.isThisValueACurrentAstCall(node):
          for target in node.targets:
            if ins(target,ast.Tuple):
              sub = 0;
              for elem in target.elts:
                new_value = ast.Subscript(value=node.value,slice=ast.Index(value=ast.Constant(value=sub)));
                new_node = ast.Assign(targets=[elem],annotation=None,value=new_value);
                self._Assign_Helper(new_node);
                sub+=1;
            else:
              self._Assign_Helper(ast.Assign(targets=[target],annotation=None,value=node.value));
 
    def _Assign_Helper(self, node):
        # targets=[target], value; can have multiple targets I guess, e.g., x = y = f()
        # print("\nAssign_Helper: ",ast.dump(node));

        # We don't care if the current WP is TRUE.
        if self.stack==[] or equality.compare(self.stack[-1],TRUE): return; # TODO: GLOBAL ASSIGNMENTS!!! or TRUE
        post = self.stack.pop();
        #if config.PRINT_DEBUG: print('In visit_Assign post is: ',ast.dump(post));
        if config.PRINT_DEBUG: print('In visit_Assign, Before: ',pprint2.pprint_top(post));
        wp = post;
        assert len(node.targets) <= 1
        for target in node.targets:        
           if config.PRINT_SOME_INFO: print('Replacing: ',ast.dump(target)," with ",ast.dump(node.value));
           # it is guaranteed that node.targets is of length 1
           # TODO: Added heuristic to NOT replace X or y lhs. 
           if (ins(target,ast.Name) and target.id=='X') or (ins(target,ast.Name) and target.id=='y'):
             continue;

           needSpecialReturn = False
           RHSisDict = False
           if isinstance(node.value, ast.Call):
             if helper.checkIfSpecial(node.value, self.helper_function):
              needSpecialReturn = True
             RHSisDict = helper.checkIfRHSDict(node.value)

           if occurs_in_formula(wp,target):
             # print('target', ast.dump(target), 'occurs in wp?');
             # TODO: Trying out unsound_tracker
             # self.unsound_Tracker(node);
             if needSpecialReturn:
              replacement = helper.getSpecialReturn(node.value, self.helper_function)
              wp = replace_in_formula(wp, target, replacement)
             elif RHSisDict:
              replacement = helper.dictToAstDict(node.value)
              if config.PRINT_SOME_INFO: print('^ ignore above, Replacing: ',ast.dump(target)," with ",ast.dump(replacement))
              wp = replace_in_formula(wp, target, replacement)
              #print(">>>>>>", pprint2.pprint_top(wp))

             else:
              #print("REPLACE!")
              #print("> before Replace:", ast.dump(wp))
              #print("> before Replace:", pprint2.pprint_top(wp))
              wp = replace_in_formula(wp,target,node.value);
              if 0:
                impl_limit = 200
                msg = "FILTERED WP. IT HAS > " + str(impl_limit) + " IMPLICATIONS AT SOME POINT."
                if helper.impl_counter(wp) >= impl_limit:
                  wp = ast.Constant(value = msg)
              #print("> after Replace:", ast.dump(wp))
              #print("> after Replace:", pprint2.pprint_top(wp))
              #print("\n\n")

           # if RHS is Tuple, try to assign each elt to LHS[0], LHS[1], ... TODO: probably need to do this on all assignment (???)
           if ins(node.value, ast.Tuple):
            num = 0
            for elt in node.value.elts:
              param_sub = ast.Subscript(value=ast.Name(id=node.targets[0].id),slice=ast.Constant(value=num))
              num += 1
              if config.PRINT_SOME_INFO: print("[Assignment - RHS is Tuple, Trying to assign each elt to LHS[n]] param: ", ast.unparse(param_sub), "with value: ", ast.dump(elt))
              if occurs_in_formula(wp, param_sub):
                wp = replace_in_formula(wp, param_sub, elt)

           # debugging code...
           if not (equality.compare(post,TRUE)) and not (ins(target,ast.Name) or (ins(target,ast.Attribute) and ins(target.value,ast.Name) and target.value.id == 'self')):
             if config.PRINT_DEBUG: print("Aliasing node? ", ast.dump(node));  
        if config.PRINT_DEBUG: print('After: ',pprint2.pprint_top(wp));

        REF_RHS = node.value
        REF_Qpre = wp
        mod_set = self.get_mod_set(REF_RHS)
        read_set = self.get_read_set(REF_Qpre)

        intersect = REF_ref_immutability.intersect_mod_read(mod_set, read_set)
        old_flag = self.curr_SN_flag
        self.curr_SN_flag = self.curr_SN_flag and (not intersect)

        if config.PRINT_DEBUG:
          print("\n======> REF IMM (Assign)")
          print("RHS: ", ast.unparse(REF_RHS))
          print("mod(RHS):")
          for loc in mod_set: print("--- ",loc) 
          #print("Qpre: ", pprint2.pprint_top(REF_Qpre))
          print("read(Qpre):")
          for loc in read_set: print("--- ",loc)
          print("intersect? ", intersect)
          print("old_SN_flag: ", old_flag)
          print("new_SN_flag: ", self.curr_SN_flag)

        self.stack.append(wp);
        assert len(self.stack) == 1
           
        if len(mod_set) != 0:
          if config.PRINT_DEBUG: 
            for loc in mod_set: print("--- ",loc) 
          assert False        

    def visit_If(self, node):
        #expr test, stmt* body, stmt* orelse
        if config.PRINT_DEBUG: print("\nIn visit_IF: ",ast.dump(node));
        post = self.stack[-1];
        S_post = self.curr_SN_flag
        if config.PRINT_DEBUG: print('IF Before: ',pprint2.pprint_top(post));
        self.Body_Helper(node.body);
        wp1 = self.stack.pop();
        S1 = self.curr_SN_flag

        self.stack.append(post);
        assert len(self.stack) == 1
        self.curr_SN_flag = S_post
        self.Body_Helper(node.orelse);
        wp2 = self.stack.pop();
        S2 = self.curr_SN_flag
        
        if config.PRINT_DEBUG:
          #print('\nBefore, wp1:', ast.dump(wp1),"... and ...",pprint2.pprint_top(wp1));
          #print('Before, wp2:', ast.dump(wp2),"... and ...",pprint2.pprint_top(wp2));
          print('\nBefore, wp1:', pprint2.pprint_top(wp1));
          #print(">> ", ast.dump(wp1))
          print('Before, wp2:', pprint2.pprint_top(wp2));
          #print(">> ", ast.dump(wp2))
          print('test is: ',ast.dump(node.test));
          print('\n');

          print("CCCCCCCCCCCCCCCCCCC ", equality.compare(wp1,wp2))

        if equality.compare(wp1,FALSE) and equality.compare(wp2,TRUE):
          # test => false and !test => true == !test 
          wp = si.negate(node.test);
        elif equality.compare(wp2,FALSE) and equality.compare(wp1,TRUE):
          # test => true and !test => false
          wp = node.test;
        elif equality.compare(wp1,wp2): 
          wp = wp1;
        elif equality.compare(wp2,FALSE):
          wp = si.cons_and(node.test,wp1); #ast.BoolOp(op=ast.And(),values=[node.test,wp1]) 
        elif equality.compare(wp1,FALSE):
          wp = si.cons_and(si.negate(node.test),wp2); 
        elif equality.compare(wp1,TRUE):
          # wp is !test => wp2 == test or wp2                                                                                              
          # wp = ast.BoolOp(op=ast.Or(),values=[node.test,wp2])
          wp = si.cons_impl(si.negate(node.test),wp2); # ast.BoolOp(op=Impl(),values=[negate(node.test),wp2])
        elif equality.compare(wp2,TRUE):         
          # test => wp1 == Not(test) or wp1                                                                          
          wp = si.cons_impl(node.test,wp1); # ast.BoolOp(op=Impl(),values=[node.test,wp1]);       
        else:       
          wp = si.factor_out(node.test,wp1,wp2);
        #if config.PRINT_DEBUG: print('\nIn visit_IF WP is: ',ast.dump(wp));
        #if config.PRINT_DEBUG: print('\nIn visit_IF WP is: ',pprint2.pprint_top(wp));
        if config.PRINT_DEBUG: print('IF After: ',pprint2.pprint_top(wp));

        REF_E = node.test
        REF_Qpre = wp
        mod_set = self.get_mod_set(REF_E)
        read_set = self.get_read_set(REF_Qpre)

        intersect = REF_ref_immutability.intersect_mod_read(mod_set, read_set)
        old_flag = S_post
        self.curr_SN_flag = S1 and S2 and (not intersect)

        if config.PRINT_DEBUG:
          print("\n======> REF IMM (IF)")
          print("E (test): ", ast.unparse(REF_E))
          print("mod(E):")
          for loc in mod_set: print("--- ",loc) 
          #print("Qpre: ", pprint2.pprint_top(REF_Qpre))
          print("read(Qpre):")
          for loc in read_set: print("--- ",loc)
          print("intersect? ", intersect)
          print("old_SN_flag: ", old_flag)
          print("S1_flag: ", S1)
          print("S2_flag: ", S2)
          print("new_SN_flag: ", self.curr_SN_flag)

        self.stack.append(wp);
        assert len(self.stack) == 1

    def visit_Raise(self, node):
      #print("\nRaise: ",ast.dump(node, include_attributes=True));
      if node == self.curr_ast_node:
        if config.PRINT_DEBUG: print("RRRRRRRRRRRRRRRR: ", ast.dump(node),"\n")
        assert equality.compare(self.wps[self.curr_ast_node_key], FALSE)
        # Soundness flag was already set to True at visit_FunctionDef
        assert self.REF_soundness_flag[self.curr_ast_node_key] == True
        self.stack.pop();
        #self.stack.append(self.wps[self.curr_ast_node]);
        self.stack.append(self.wps[self.curr_ast_node_key])
        assert len(self.stack) == 1
        self.curr_SN_flag = True

    # requires: body is a body, i.e., list of stmts                                                                                            
    def Body_Helper(self, body):
      for stmt in reversed(body):
        #print("\n!!!!!!! Here the stmt is:", ast.unparse(stmt),"\n")
        #print("[[[[[[ current WP is ", pprint2.pprint_top(self.stack[-1]))
        super(Analyzer, self).visit(stmt);

    def visit_Return(self, node):
      # return func_call(...))
      # Can we just visit the func_call()?
      if ins(node.value, ast.Call):
        call_return = node.value
        super(Analyzer, self).visit(call_return)
      else:
        super(Analyzer, self).generic_visit(node);
        self.stack.pop();
        self.stack.append(TRUE);
        assert len(self.stack) == 1
        self.curr_SN_flag = True # ??? 
      # if (node.value != None): self.visit(node.value);

    # Checking unsoundness 

    def visit_For(self, node):
        #print("\nFor: ",ast.dump(node)); 
        # TODO: Needs testing!!!
        # Targets, Pass, Break, Continue
        if equality.compare(self.stack[-1],TRUE):
          if_test = ast.Compare(left=node.target, ops=[ast.In()], comparators=[node.iter])
          self.visit_If(ast.If(test=if_test,body=node.body,orelse=node.orelse))
          REF_E1E2 = if_test
          REF_Qpre = self.stack[-1]
          mod_set = self.get_mod_set(REF_E1E2)
          read_set = self.get_read_set(REF_Qpre)

          intersect = REF_ref_immutability.intersect_mod_read(mod_set, read_set)
          old_flag = self.curr_SN_flag
          self.curr_SN_flag = self.curr_SN_flag and (not intersect)

          if config.PRINT_DEBUG:
            print("\n======> REF IMM (For_Qpost == True)")
            print("E1 in E2: ", ast.unparse(REF_E1E2))
            print("mod(E1 in E2):")
            for loc in mod_set: print("--- ",loc) 
            #print("Qpre: ", pprint2.pprint_top(REF_Qpre))
            print("read(Qpre):")
            for loc in read_set: print("--- ",loc)
            print("intersect? ", intersect)
            print("old_SN_flag: ", old_flag)
            print("new_SN_flag: ", self.curr_SN_flag)

        else:
          self.unsound_Tracker(node)
          REF_FOR = node
          REF_Qpost = self.stack[-1]
          mod_set = self.get_mod_set(REF_FOR)
          read_set = self.get_read_set(REF_Qpost)

          intersect = REF_ref_immutability.intersect_mod_read(mod_set, read_set)
          old_flag = self.curr_SN_flag
          self.curr_SN_flag = self.curr_SN_flag and (not intersect)

          if config.PRINT_DEBUG:
            print("\n======> REF IMM (For_Qpost != True)")
            print("For: ", ast.unparse(REF_FOR))
            print("mod(For):")
            for loc in mod_set: print("--- ",loc) 
            print("Qpost: ", pprint2.pprint_top(REF_Qpost))
            print("read(Qpost):")
            for loc in read_set: print("--- ",loc)
            print("intersect? ", intersect)
            print("old_SN_flag: ", old_flag)
            print("new_SN_flag: ", self.curr_SN_flag)

        if 0:
          if len(mod_set) != 0:
            for loc in mod_set: print("--- ",loc) 
            assert False  


    def visit_AsynchFor(self, node):
        self.unsound_Tracker(node);
        self.unsoundCheck(node)
    
    def visit_Delete(self, node):
        self.unsound_Tracker(node);
        self.unsoundCheck(node)

    def visit_With(self, node):
        #print("\nWith: ",ast.dump(node));
        self.Body_Helper(node.body) 
        #self.unsound_Tracker(node);

    def visit_AsynchWith(self, node):
        self.unsound_Tracker(node);
        self.unsoundCheck(node)
    
    def visit_Expr(self, node):
        super(Analyzer, self).generic_visit(node);
                
    def visit_Continue(self, node):
        pass; # TODO: continue is TRUE? 

    def unsoundCheck(self, node):
        REF_Other = node
        REF_Qpost = self.stack[-1]
        mod_set = self.get_mod_set(REF_Other)
        read_set = self.get_read_set(REF_Qpost)

        intersect = REF_ref_immutability.intersect_mod_read(mod_set, read_set)
        old_flag = self.curr_SN_flag
        self.curr_SN_flag = self.curr_SN_flag and (not intersect)

        if config.PRINT_DEBUG:
          print("\n======> REF IMM (Other)")
          print("Other: ", ast.unparse(REF_Other))
          print("mod(Other):")
          for loc in mod_set: print("--- ",loc) 
          print("Qpost: ", pprint2.pprint_top(REF_Qpost))
          print("read(Qpost):")
          for loc in read_set: print("--- ",loc)
          print("intersect? ", intersect)
          print("old_SN_flag: ", old_flag)
          print("new_SN_flag: ", self.curr_SN_flag)      

    def getKwarg(self, node):
      if self.kwargName is None:
        if self.function_map[self.func_name].args.kwarg:
          #print(">>>>>>>>>>>>>>>>>>>>>>>>>>HERE AT containKwarg")
          #print(self.func_name)
          #print(ast.dump(self.function_map[self.func_name].args.kwarg))
          self.kwargName = self.function_map[self.func_name].args.kwarg.arg
      return self.kwargName

    def visit_Call(self, node):
        super(Analyzer, self).generic_visit(node);
        #print(ast.dump(node))
        #print(ast.dump(self.curr_ast_node))
        if node == self.curr_ast_node:
          if config.PRINT_DEBUG: print(">>> IN Call AST")
          #print("___",pprint2.pprint_top(self.stack[-1]))
          wp_before = self.stack[-1]
          self.stack.pop()
          self.stack.append(self.wps[self.curr_ast_node_key])
          self.curr_SN_flag = self.REF_soundness_flag[self.curr_ast_node_key]
          assert len(self.stack) == 1
        # self.unsound_Tracker(node);
        # print("\nCall: In the process of figuring out...", ast.dump(node.func)); #,ast.dump(node));
          self.Call_Helper(node, wp_before, self.curr_ast_node_key[-1]);
        else:
          self.unsoundCheck(node) #TODO: unsound check here ?????????????????

    def Call_Helper(self, node, wp_before, callee):

        callee_F_arg = self.function_map[callee].args
        self.kwargName = self.getKwarg(node)
        if self.kwargName is not None:
          if config.PRINT_DEBUG: print("KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK ", self.kwargName)

        if config.PRINT_SOME_INFO:
          print("\n   Call_Helper ");
          print("<< AST node:",ast.dump(node))
        #if node.keywords[0].arg is None:
        #  print("NONEEEEEEEEEEEE")
        #  print(ast.dump(node.keywords[0]))
        #  print(node.keywords[0].value.id)
        
        #print("F params:",ast.dump(self.params)) # probably wont be used
        if config.PRINT_SOME_INFO: print("<< Callee name:", callee)
        if callee == "/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/sklearn/metrics/pairwise.py:None:pairwise_kernels":
          self.stack.pop()
          self.stack.append(ast.Constant(value="SKIP THE CALL TO /metrics/pairwise.py:None:pairwise_kernels"))
          assert len(self.stack) == 1
          return
        #TODO: Check this
        if 0:
          if len(callee.split(":")) > 3:
            callee_F_arg = self.function_map[callee.replace(":"+callee.split(":")[-1],"")].args
            print(callee.replace(":"+callee.split(":")[-1],""))
            #assert False

        # TODO: WE PROBABLY WANT TO PRINT THIS MOST OF THE TIME
        if 0:
          print("<< Callee params:",ast.dump(callee_F_arg))
          print("<< Current wp in stack:", ast.dump(self.stack[-1]))
          print("<< Current soundness flag:", self.curr_SN_flag)
          print("vararg:", ast.dump(callee_F_arg.vararg)) if callee_F_arg.vararg is not None else print("vararg: None")
          print("kwarg:", ast.dump(callee_F_arg.kwarg)) if callee_F_arg.kwarg is not None else print("kwarg: None")

        func_wp = self.stack[-1]
        if 0:
          print("Before: ", pprint2.pprint_top(func_wp))

        # Check if this call pass kwarg
        thisCallPassKwarg = False
        kwargArgName = "" # There are cases like X = check_array(X, **check_X_params)         y = check_array(y, **check_y_params)
                          # Caller's kwarg and Callee's (**kwargs) does NOT have to be the same name
        for keyword in node.keywords:
          if keyword.arg is None:
            #assert keyword.value.id == self.kwargName # if keyword.arg is None then it must be (**kwargs) right???
            kwargArgName = keyword.value.id
            thisCallPassKwarg = True

        #####################
        # Add to defaultDict
        if thisCallPassKwarg:
          if callee_F_arg.kwarg: # Callee also has (**kwargs)
            # PANIC
            # print("\n",self.kwargDefault,"\n")
            # If func_wp is TRUE then we let it go
            if equality.compare(func_wp,TRUE):
              if config.PRINT_SOME_INFO: print("After (Caller and Callee has (**kwargs) but WP is TRUE): ", pprint2.pprint_top(func_wp))
              self.stack.pop()
              if equality.compare(wp_before,TRUE):
                self.stack.append(func_wp)
              else:
                self.stack.append(ast.BoolOp(op=ast.And(),values=[wp_before,func_wp]))
              assert len(self.stack) == 1
              return 

            func_wp = ast.Constant(value="[TODO]Caller and Calle both have **kwargs. Dont support this right now.")
            #assert False, "Callee also has (**kwargs). Another case."

          else:
            defaultDict = {}
            # Add default values of keyword-only arguments
            if len(callee_F_arg.kw_defaults) != 0:
              assert len(callee_F_arg.kw_defaults) == len(callee_F_arg.kwonlyargs)
              count = 0
              for kw in callee_F_arg.kwonlyargs:
                defaultDict[kw.arg] = callee_F_arg.kw_defaults[count].value
                count += 1

            # Add default values of arguments that can be passed positionally
            de_pos = len(callee_F_arg.defaults) - 1
            arg_pos = len(callee_F_arg.args) - 1
            num_pos_arg = len(node.args) # number of positional arg of the call 
            while de_pos >= 0:# and arg_pos >= num_pos_arg:
              #print("OOOO", de_pos, arg_pos, num_pos_arg)
              defaultDict[callee_F_arg.args[arg_pos].arg] = callee_F_arg.defaults[de_pos].value
              de_pos -= 1
              arg_pos -= 1 

            # Replace default values with keyword values if applicable
            #for keyword in node.keywords:
            #  if keyword.arg in defaultDict:
            #   defaultDict[keyword.arg] = keyword.value

            # ACTUALLY, we can safely remove them from the dict and just do the replacement right?
            # We remove here, then do the replacement after pos arg
            for keyword in node.keywords:
              if keyword.arg in defaultDict:
                defaultDict.pop(keyword.arg, None)

            # Remove positional argument (arg) from the dict
            num = 0
            for arg in node.args:
              if ins(arg, ast.Name):
                if config.PRINT_SOME_INFO: print("REMOVE", ast.dump(arg), callee_F_arg.args[num].arg)
                defaultDict.pop(callee_F_arg.args[num].arg, None) #has to be name from callee
              else:
                defaultDict.pop(callee_F_arg.args[num].arg, None)
                #assert False, "What else?"
              num += 1

            # In case a function has multiple calls with diff kwarg. store as [k1, dict, k2, dict2, ...]
            if 0:
              if self.curr_ast_node_key not in self.kwargDefault:
                self.kwargDefault[self.curr_ast_node_key] = [kwargArgName, defaultDict]
              else:
                self.kwargDefault[self.curr_ast_node_key].append(kwargArgName)
                self.kwargDefault[self.curr_ast_node_key].append(defaultDict)

            #print(defaultDict)
            ####
            # Positional arguments
            num = 0
            if len(callee_F_arg.args) != 0:
              if callee_F_arg.args[0].arg == "self":
                num = 1
              for arg in node.args:
                param = ast.Name(id=callee_F_arg.args[num].arg)
                if config.PRINT_SOME_INFO: print("\t[thisCallPassKwarg - pos] param: ", param.id + ", with arg: ", ast.dump(arg))
                if occurs_in_formula(func_wp, param):
                  func_wp = replace_in_formula(func_wp, param, arg)
                num += 1

            # Replace dict with i.e. "check_params["dtype"]"
            for key in defaultDict.keys():
              param = ast.Name(id=key)
              if config.PRINT_SOME_INFO: print("\t[thisCallPassKwarg - substitute]: ", param.id, "with", kwargArgName+".get(\""+param.id+"\","+helper.ToStr(defaultDict[key])+")")
              #print(defaultDict[key], ast.dump(helper.ToAst(defaultDict[key])))
              subs = ast.Call(func=ast.Attribute(value=ast.Name(id=kwargArgName), attr='get'), 
                      args=[ast.Constant(value=param.id), helper.ToAst(defaultDict[key])], keywords=[])
              #subs = ast.Subscript(value=ast.Name(id=kwargArgName), slice=ast.Name(id=param.id))
              if occurs_in_formula(func_wp, param):
                #func_wp = replace_in_formula(func_wp, param, subs, getkwargDefaultList(self.kwargDefault))
                func_wp = replace_in_formula(func_wp, param, subs)            

            # Replace default for pos arg.
            # ^ We DONT do this here because it can be part of kwarg down the path (eg. accept_sparse). We added them in defaultDict above.

            # Replace default for keyword arg. They wont be in kwarg.
            for keyword in node.keywords:
              if keyword.arg is not None:
                param = ast.Name(id=keyword.arg)
                if config.PRINT_SOME_INFO: print("\t[thisCallPassKwarg - keyword default]: ", param.id, "with arg: ", ast.dump(keyword.value))
                if occurs_in_formula(func_wp, param):
                  func_wp = replace_in_formula(func_wp, param, keyword.value)               

            #print(defaultDict)
            #print("\nAfter[1111]: ", pprint2.pprint_top(func_wp))

        # This call does not pass kwarg (no **xyz as parameter)
        # We have 2 cases; whether Callee has kwarg (which is for **kwargs)
        else:
          # The caller does not pass kwarg, the callee has kwarg. 
          # Time to substitute from dict. 
          if callee_F_arg.kwarg:
            # keywords in **kwargs are those in (node.keywords) - those in (callee_F_arg.args AND .kwonlyargs)
            actual_kwargs_dict = {}
            for keyword in node.keywords:
              actual_kwargs_dict[keyword.arg] = keyword.value
            #print("\n",actual_kwargs_dict)
            
            for arg in callee_F_arg.args:
              actual_kwargs_dict.pop(arg.arg, None)
            #print("\n",actual_kwargs_dict)
            
            for kw in callee_F_arg.kwonlyargs:
              actual_kwargs_dict.pop(kw.arg, None)
            if config.PRINT_SOME_INFO: print("\nactual_kwargs_dict: ",actual_kwargs_dict)

            # replace this dict to *kwargs. It should result in something like {ThisDict}.get('ABC', value)
            tar = ast.Name(id=callee_F_arg.kwarg.arg)
            rep_dict = ast.Dict(keys=[], values=[])
            for k in actual_kwargs_dict.keys():
              rep_dict.keys.append(ast.Constant(value=k))
              rep_dict.values.append(actual_kwargs_dict[k])

            if rep_dict is None:
              assert False, "Replacement dict is None ???"

            if occurs_in_formula(func_wp, tar):
              if config.PRINT_SOME_INFO: print("\t[CalleeHasKwarg - replace kwarg with actual dict and simplify (if possible)] kwarg: " + tar.id + " ,dict", pprint2.pprint_top(rep_dict))
              func_wp = replace_in_formula(func_wp, tar, rep_dict)

            # Replace postional arguments
            num = 0
            pos_arg = []
            if len(callee_F_arg.args) != 0:
              if callee_F_arg.args[0].arg == "self":
                num = 1
              for arg in node.args:
                if num >= len(callee_F_arg.args): #SVC
                  continue
                param = ast.Name(id=callee_F_arg.args[num].arg)
                if config.PRINT_SOME_INFO: print("\t[CalleeHasKwarg - pos] param: ", param.id + ", with arg: ", ast.dump(arg))
                pos_arg.append(param.id)
                if occurs_in_formula(func_wp, param):
                  func_wp = replace_in_formula(func_wp, param, arg)
                num += 1   

            # Default values for positional arg
            # They are in "args" and default values are in "defaults".
            # "If there are fewer defaults, they correspond to the last n arguments.""
            # However, they can be passed with keyword too so we have to check at node.keywords too            
            de_pos = len(callee_F_arg.defaults) - 1
            arg_pos = len(callee_F_arg.args) - 1
            num_pos_arg = len(node.args) # number of positional arg of the call 

            while de_pos >= 0:# and arg_pos >= num_pos_arg:
              # check if this arg already have value from keyword arg
              found = False
              for keyword in node.keywords:
                if keyword.arg == callee_F_arg.args[arg_pos].arg:
                  found = True
              # check if this arg get passed with var/value
              if callee_F_arg.args[arg_pos].arg in pos_arg:
                found = True
              # replace with default value
              if not found:
                param = ast.Name(id = callee_F_arg.args[arg_pos].arg)
                if config.PRINT_SOME_INFO: print("\t[CalleeHasKwarg - pos default] param:", param.id, "with arg: ", ast.dump(callee_F_arg.defaults[de_pos]))
                if occurs_in_formula(func_wp, param):
                  if config.PRINT_SOME_INFO: print("Occur = YES")
                  func_wp = replace_in_formula(func_wp, param, callee_F_arg.defaults[de_pos])
              de_pos -= 1
              arg_pos -= 1 

            # Replace keyword arguments. Those in (node.keywords) - those in (actual_kwargs_dict)[because they are in kwarg).
            for keyword in node.keywords:
              if keyword.arg not in actual_kwargs_dict:
                param = ast.Name(id=keyword.arg)
                if config.PRINT_SOME_INFO: print("\t[CalleeHasKwarg - keyword] param: ", param.id, "with value: ", ast.dump(keyword.value))
                if occurs_in_formula(func_wp, param):
                  func_wp = replace_in_formula(func_wp, param, keyword.value)

                # if RHS is Tuple, try to assigne each elt to LHS[0], LHS[1], ... TODO: probably need to do this on all assignment (???)
                if ins(keyword.value, ast.Tuple):
                  num = 0
                  for elt in keyword.value.elts:
                    param_sub = ast.Subscript(value=ast.Name(id=keyword.arg),slice=ast.Constant(value=num))
                    num += 1
                    #print(ast.dump(param_sub), ast.dump(elt))
                    if config.PRINT_SOME_INFO: print("\t[CalleeHasKwarg - keyword RHS Tuple] param: ", ast.unparse(param_sub), "with value: ", ast.dump(elt))
                    if occurs_in_formula(func_wp, param_sub):
                      func_wp = replace_in_formula(func_wp, param_sub, elt)

            # Replace callee's default keyword values
            if len(callee_F_arg.kw_defaults) != 0:
              assert len(callee_F_arg.kw_defaults) == len(callee_F_arg.kwonlyargs)
              count = 0
              for kw in callee_F_arg.kwonlyargs:
                found = False
                for keyword in node.keywords:
                  if keyword.arg == kw.arg:
                    found = True
                # check if this arg get passed with var/value. Should not be the case here    
                if kw.arg in pos_arg:
                  found = True
                  assert False, "Should not be the case. Check just for sure"
                # replace with default value
                if not found:
                  if config.PRINT_SOME_INFO: print("\t[CalleeHasKwarg - keyword default] param: ", kw.arg, "with arg: ", ast.dump(callee_F_arg.kw_defaults[count]))
                  if callee_F_arg.kw_defaults[count] is not None:
                    param = ast.Name(id = kw.arg)
                    if occurs_in_formula(func_wp, param):
                      func_wp = replace_in_formula(func_wp, param, callee_F_arg.kw_defaults[count])
                count += 1


            if config.PRINT_DEBUG: print("\n WIP wp:",pprint2.pprint_top(func_wp),"\n")

          # The caller does NOT pass kwarg, the callee does NOT have kwarg.
          # Normal case i guess
          else:
            # Just do the normal positional arg and keyword arg 
            num = 0
            pos_arg = []
            if len(callee_F_arg.args) != 0:
              if callee_F_arg.args[0].arg == "self":
                num = 1
              if 0:
                for aa in node.args:
                  print(ast.dump(aa))
                print("\n")
                for aa in callee_F_arg.args:
                  print(ast.dump(aa))
                print("\n")
              for arg in node.args:
                # HEREE
                if num >= len(callee_F_arg.args): # ridge
                  break
                param = ast.Name(id=callee_F_arg.args[num].arg)
                if config.PRINT_SOME_INFO: print("   [normal - pos] param: ", param.id, "with arg: ", ast.dump(arg))
                pos_arg.append(param.id)
                if occurs_in_formula(func_wp, param):
                  func_wp = replace_in_formula(func_wp, param, arg)
                num += 1

            # Default values for positional arg
            # They are in "args" and default values are in "defaults".
            # "If there are fewer defaults, they correspond to the last n arguments.""
            # However, they can be passed with keyword too so we have to check at node.keywords too
            de_pos = len(callee_F_arg.defaults) - 1
            arg_pos = len(callee_F_arg.args) - 1
            num_pos_arg = len(node.args) # number of positional arg of the call 

            while de_pos >= 0: # and arg_pos >= num_pos_arg:
              # check if this arg already have value from keyword arg
              found = False
              for keyword in node.keywords:
                if keyword.arg == callee_F_arg.args[arg_pos].arg:
                  found = True
              # check if this arg get passed with var/value
              if callee_F_arg.args[arg_pos].arg in pos_arg:
                found = True
              # replace with default value
              if not found:
                param = ast.Name(id = callee_F_arg.args[arg_pos].arg)
                if config.PRINT_SOME_INFO: print("   [normal pos - default] param: ", param.id, "with arg: ", ast.dump(callee_F_arg.defaults[de_pos]))
                if occurs_in_formula(func_wp, param):
                  func_wp = replace_in_formula(func_wp, param, callee_F_arg.defaults[de_pos])
              de_pos -= 1
              arg_pos -= 1 

            # keyword arg 
            for keyword in node.keywords:
              if keyword.arg is not None: # If None then it's (**kwargs)
                param = ast.Name(id=keyword.arg)
                if config.PRINT_SOME_INFO: print("   [normal keyword] param: ", param.id,"with arg: ", ast.dump(keyword.value))
                if occurs_in_formula(func_wp,param):
                  func_wp = replace_in_formula(func_wp, param, keyword.value)

            # Default values for keyword arg
            if len(callee_F_arg.kw_defaults) != 0:
              assert len(callee_F_arg.kw_defaults) == len(callee_F_arg.kwonlyargs)
              count = 0
              for kw in callee_F_arg.kwonlyargs:
                found = False
                for keyword in node.keywords:
                  if keyword.arg == kw.arg:
                    found = True
                # check if this arg get passed with var/value. Should not be the case here    
                if kw.arg in pos_arg:
                  found = True
                  assert False, "Should not be the case. Check just for sure"
                # replace with default value
                if not found:
                  if config.PRINT_SOME_INFO: print("   [normal kw_defaults] param: ", kw.arg, "with arg: ", ast.dump(callee_F_arg.kw_defaults[count]))
                  if callee_F_arg.kw_defaults[count] is not None:
                    param = ast.Name(id = kw.arg)
                    if occurs_in_formula(func_wp, param):
                      func_wp = replace_in_formula(func_wp, param, callee_F_arg.kw_defaults[count])
                count += 1


        if config.PRINT_DEBUG: print("After: ", pprint2.pprint_top(func_wp),"\n")
        self.stack.pop()
        if equality.compare(wp_before,TRUE):
          self.stack.append(func_wp)
          # TODO do we need to adjust soundness flag here?
        else:
          assert False, "PANIC"
          self.stack.append(ast.BoolOp(op=ast.And(),values=[wp_before,func_wp])) 
        assert len(self.stack) == 1
        return

    # Safe to skip Expr and Delete

    def unsound_Tracker(self, node):
       if not equality.compare(self.stack[-1],TRUE):
          self.unsound_stack.pop();
          self.unsound_stack.append(True);
          #print("\n Unsound stmt ",ast.dump(node),"\n");


# Right now main is not used. The expected usage of wp_intra is 
# to call visit_FunctionDef on a FunctionDef node.

