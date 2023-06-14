#!/usr/local/bin/python3.9

import copy
import ast
import sys
import os

import config
import graph
import crawler

def ins(obj,cls):
  return isinstance(obj,cls);

# required ins(node,ast.Call)
def ok_receiver(node):
  func = node.func
  if (ins(func,ast.Attribute) and ins(func.value,ast.Name) and func.value.id == 'np'):
    return True
  if (ins(func,ast.Attribute) and ins(func.value,ast.Name) and func.value.id == 'sp'):
    return True
  if (ins(func,ast.Attribute) and ins(func.value,ast.Name) and func.value.id == 'warnings'):
    return True
  if (ins(func,ast.Attribute) and ins(func.value,ast.Name) and func.value.id == 'array'):
    return True
  if (ins(func,ast.Attribute) and ins(func.value,ast.Constant)):
    return True
  return False

def ok_name(func_name):
  ok_names = ['TypeError','RuntimeError','ValueError','min','max','isinstance','int','ceil','super','len','issparse','getattr','hasattr','callable','str','reversed']
  added_names = ["type","print"]
  ok_names += added_names
  return func_name in ok_names;

class CallGraphAnalyzer(ast.NodeVisitor):
  
  def __init__(self,main,main_class,function_map,bases_map):
    self.call_graph = graph.Graph();
    self.visited = [];
    self.main_func = main
    self.main_class = main_class
    self.curr_func = ""
    self.worklist = []
    self.function_map = function_map
    self.bases_map = bases_map
    #self.function_map, self.bases_map = crawler.get_function_map()
    self.curr_call_node = None

  def visit_Call(self,node):
    # Call(expr func, expr* args, keyword* keywords)
    if config.PRINT_DEBUG: print("!!! Visiting call: ", ast.dump(node))
    self.curr_call_node = node
    if ins(node.func,ast.Name):
      if config.PRINT_DEBUG: print("Function is a Name.")
      func_list = self.resolve_name(node.func.id)
      for func_name in func_list:
        # print("Call: ", ast.dump(node))
        # print('IN NAME Func call from ',self.curr_func," to ",func_name);
        self.new_edge(self.curr_func,func_name,node)
      if func_list == [] and not ok_name(node.func.id):
        if config.PRINT_DEBUG: 
          print("Call: ", ast.dump(node))
          print("NAME Function call in ",self.curr_func," is something else...\n")
          print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    elif (ins(node.func,ast.Attribute) and ins(node.func.value,ast.Name) and self.resolve_base(node.func.value.id) and len(node.args)>0 and ins(node.args[0],ast.Name) and node.args[0].id == 'self'):
      #print("Call: ", ast.dump(node)) 
      if config.PRINT_DEBUG: print("Function call is DIRECT call ClassName.methodname(self)")
      func_name = self.resolve_self(node.func.attr,self.resolve_base(node.func.value.id))
      if config.PRINT_DEBUG: print("2 func_name",func_name)
      if func_name in self.function_map.keys(): 
        self.new_edge(self.curr_func,func_name,node)
      else:
        if config.PRINT_DEBUG: 
          print("Call: ", ast.dump(node))
          print("Class.func(self) in ",self.curr_func," is something else...\n")
          print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        #sys.exit(0)
    elif (ins(node.func,ast.Attribute) and ins(node.func.value,ast.Name) and node.func.value.id == 'self'):
      if config.PRINT_DEBUG: print("Function call is 3")
      func_name = self.resolve_self(node.func.attr,self.main_class)
      if config.PRINT_DEBUG: print("3 func_name = ", func_name)
      if func_name in self.function_map.keys(): 
        self.new_edge(self.curr_func,func_name,node);
      else:
        if config.PRINT_DEBUG: 
          print("Call: ", ast.dump(node))
          print("self.func in ",self.curr_func," is something else...\n")
          print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    elif (ins(node.func,ast.Attribute) and ins(node.func.value,ast.Call) and ins(node.func.value.func,ast.Name) and node.func.value.func.id=='super'):
      if config.PRINT_DEBUG: print("Function call is 4")
      curr_func_split = self.curr_func.split(':')
      #print("curr_func_split ",curr_func_split)
      #for b in self.bases_map:
      #  print(b)
      encl_class = curr_func_split[0]+":"+curr_func_split[1]
      bases = self.bases_map[encl_class]
      #print("bases ",bases)
      for base in reversed(bases):
        full_base = self.resolve_base(base)
        #print("base: ",full_base)
        func_name = self.resolve_self(node.func.attr,full_base);
        if not (func_name == None):
          # print("Call: ", ast.dump(node))
          # print("Function call in ",self.curr_func,"...")
          self.new_edge(self.curr_func,func_name,node)
          break;
      #TODO: Fall-through clause if func not found?
    elif ok_receiver(node):
      # print("Numpy call or warnings call or else")
      pass
    else:
      # ex1: kwargs 
      # def _validate_data(self, ... , **check_params):
      #   check_params.get('ensure_2d', True)
      # ex2: 
      # from scipy import linalg
      # ... = linalg.svd(...)
      # ex3:
      # X_var = ((X.multiply(X)).mean() - (X.mean()) ** 2

      if ins(node.func,ast.Attribute) and ins(node.func.value,ast.Name):
        if config.PRINT_DEBUG: print("AEIOU ",self.resolve_base(node.func.value.id))
        #func_name = self.resolve_self(node.func.attr,self.resolve_base(node.func.value.id))
        #print("func_name ", func_name)
        #print("in? ", func_name in self.function_map.keys())

      if config.PRINT_DEBUG: 
        print("Call: ", ast.dump(node))
        print("Function call in ",self.curr_func," is something else...\n")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    super(CallGraphAnalyzer, self).generic_visit(node);    

  def new_edge(self,caller,callee,node):
    self.call_graph.addEdge(graph.Edge(caller,callee,node))
    if callee not in self.visited:
      self.visited.append(callee)
      self.worklist.append(callee)

  def solve_worklist(self):
    self.visited.append(self.main_func)
    self.worklist.append(self.main_func)
    while not self.worklist == []:
      func_name = self.worklist.pop(0)
      func_ast = self.function_map[func_name]
      self.curr_func = func_name

      if config.PRINT_DEBUG: print("\n >>> WORKLIST now at: ", func_name)
      #print(ast.dump(func_ast))
      super(CallGraphAnalyzer, self).generic_visit(func_ast)
      
  def resolve_name(self,the_name):
    func_list = []
    for full_func_name in self.function_map.keys():
      if the_name == "OrdinalEncoder":
        if config.PRINT_DEBUG: print("> ", full_func_name)
      full_func_name_split = full_func_name.split(':')
      if full_func_name_split[2] == the_name:
        func_list.append(full_func_name)
    if config.PRINT_DEBUG: print("resolve_name: func_list = ",func_list)

    #TODO: 
    return func_list

  def resolve_self(self,the_name,curr_class):
    # print("curr_class: ",curr_class)
    # TODO: Check Semantics of Inheritance!!!
    if curr_class is None:
      return None
    full_name = curr_class+":"+the_name
    #print("resolve_self: full_name = ",full_name)
    if full_name in self.function_map.keys():
      return full_name
    else:
      bases = self.bases_map[curr_class]
      #print("bases:",bases)
      for base in reversed(bases):
        full_base = self.resolve_base(base)
        #print("full_base:",full_base)
        if full_base:
          full_name = self.resolve_self(the_name,full_base)
          if full_name: 
            return full_name
      return None
    
  def resolve_base(self,the_base_name):
    for base in self.bases_map.keys():
      #print("base ", base, ": the_base_name", the_base_name)
      if base.split(':')[1] == the_base_name: 
      # TODO: what if there are more than one classes named Base???
        return base
    return None

def reverseGraph(g):
    rvG = graph.Graph()
    for key in g.edges.keys():
        for edge in g.edges[key]:
            rvG.addEdge(graph.Edge(edge.tgt,edge.src,edge.label))
    return rvG
 
def main(package_dir, class_name, function_name):

    #Analyzer takes main class, e.g., PCA, and main method, e.g., fit and constructs a call graph starting at main method.
    # for WINDOWS "C:" is removed at the beginning of the path to avoid Split() issue
    operator_main_func = ""
    operator_main_class = ""
    nameSeparator = ":" + function_name

    function_map, bases_map = crawler.get_function_map(package_dir)

    op = class_name
    ll = []
    kk = []

    for ff in function_map.keys():
      gg = ff.split(":")
      if op == gg[1] and function_name == gg[2]:
        ll.append(ff)
      if op == gg[1]:
        kk.append(ff)

    if len(ll) == 1:
      operator_main_func = ll[0]
      #operator_main_class = ll[0].replace(":fit","")
      operator_main_class = ll[0].replace(nameSeparator,"")
    else:
      if config.PRINT_DEBUG: 
        print("BUG?")
        print(ll,'\n')
        print(kk)
      if len(ll) == 0: #fit() is inheritted
        class_name = [op]
        found = False
        base_fit = ""
        while len(class_name) > 0:
          if found:
            break
          now = class_name[0]
          #print("Now =", now)
          class_name.remove(now)
          for base in bases_map.keys():
            if base.split(":")[1] == now:
              if base+nameSeparator in function_map:
                found = True
                base_fit = base+nameSeparator
              bases = reversed(bases_map[base])
              for b in bases:
                class_name.append(b)

      if found:
        #print("Base_fit =",base_fit)
        operator_main_func = base_fit
        operator_main_class = kk[0].replace(":"+kk[0].split(":")[2],"")
      else:
        found = found
        assert False, "Could not resolve function name"

    if config.PRINT_DEBUG: 
      print("\noperator_main_func:", operator_main_func)
      print("operator_main_class:", operator_main_class)

    if op == "CUSTOM":
        operator_main_func = "/PATH_TO/site-packages/sklearn/utils/validation.py:None:_check_large_sparse"
        operator_main_class = "/PATH_TO/site-packages/sklearn/utils/validation.py:None"


    analyzer = CallGraphAnalyzer(operator_main_func,operator_main_class,function_map,bases_map)
    analyzer.solve_worklist()
    analyzer.call_graph.isDAG()
    analyzer.call_graph.isDAG2()

    if config.PRINT_DEBUG: 
      analyzer.call_graph.printGraph()
      print("operator_main_func:", operator_main_func)
      print("operator_main_class:", operator_main_class)
      print("A DAG: ",analyzer.call_graph.isDAG())
      #print("A DAG2 (NOT consider self-loop): ",analyzer.call_graph.isDAG2())

    return analyzer


if __name__ == "__main__":
  print("> call_graph.py: NOTHING IS HERE")

    