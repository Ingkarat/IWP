#!/usr/local/bin/python3.9

import ast
import sys
import os

#import crawler

def ins(obj,cls):
  return isinstance(obj,cls);

class SymTableAnalyzer(ast.NodeVisitor):
  
  def __init__(self): 
    self.sym_table = {}
  
  def visit_AugAssign(self, node):
    # This is target += value 
    # TODO: Needs testing!
    # print("\nAugAssign: ",ast.dump(node, include_attributes=True)); 
    self._Assign_Helper(node.target,ast.BinOp(left=node.target,right=node.value,op=node.op));
    super(SymTableAnalyzer, self).generic_visit(node);
    
  def visit_AnnAssign(self, node):
    # print("\nAnnAssign: ",ast.dump(node, include_attributes=True)); 
    if node.value != None:
      self._Assign_Helper(node.target,node.value);
    super(SymTableAnalyzer, self).generic_visit(node);

  def visit_Assign(self, node):
    # targets, value; can have tuple, e.g., x,y = f() and multiple targets x = y = f()  
    # print("\nAssign: ",ast.dump(node));
    # self.visit(node.value);
    # print("Here in Assign ",ast.dump(node))
    for target in node.targets:
      if ins(target,ast.Tuple):
        sub = 0;
        for elem in target.elts:
          new_value = ast.Subscript(value=node.value,slice=ast.Index(value=ast.Constant(value=sub)));
          self._Assign_Helper(elem,new_value);
          sub+=1;
      else:
        self._Assign_Helper(target,node.value);
    super(SymTableAnalyzer, self).generic_visit(node); 
  
  def _Assign_Helper(self, lhs, rhs):
    # store in symbol table only if lhs is a Name or an Attribute self.attr
    # print("Here in Assign Helepr recording lhs ",ast.dump(lhs))
    if ins(lhs,ast.Name) or (ins(lhs,ast.Attribute) and ins(lhs.value,ast.Name) and lhs.value.id=="self"):
      lhs_str = ast.unparse(lhs)
      if lhs_str in self.sym_table.keys():
        self.sym_table[lhs_str].append(rhs) #= None # TODO: Check this out. Here will add to the list rather than have one iterm...
      else:
        self.sym_table[lhs_str] = [rhs] 
    else:
        print("BAD ASSIGN: ",ast.unparse(lhs)," = ",ast.unparse(rhs))

class ClosureAnalyzer(ast.NodeVisitor):
  
  # Takes an AST node and a symbol table and returns a list of Names (strings) or self.attr 
  # that have been defined outside. Some are global values?
  def __init__(self,sym_table,start_node,globals_map):
    self.sym_table = sym_table
    self.wl = []
    self.visited = []
    self.result = []
    self.start_node = start_node
    self.globals_map = globals_map

  def solve_wl(self):
    self.wl.append(self.start_node)
    while not self.wl == []:
      curr_node = self.wl.pop()
      # print("popping current node: ",ast.unparse(curr_node),ast.dump(curr_node))
      super(ClosureAnalyzer, self).visit(curr_node);      

  def visit_Name(self, node):
    # print("Processing Name: "+ast.unparse(node));
    if ast.unparse(node) in self.visited: return
    self.visited.append(node.id)
    self._process_Name(node.id)

  def visit_Attribute(self, node):
    # print("Processing Attribute: "+ast.unparse(node));
    if ast.unparse(node) in self.visited: return
    if ins(node.value,ast.Name) and node.value.id == 'self':
      self.visited.append(ast.unparse(node))
      self._process_Name(ast.unparse(node))
    else:
      if node.attr in self.globals_map.keys():
        if node.attr not in self.visited:
          self.visited.append(node.attr)
          self._process_Name(node.attr)
      elif node.attr not in self.result: self.result.append(node.attr)
      super(ClosureAnalyzer, self).visit(node.value);   

  def _process_Name(self, name):
    if name in self.sym_table.keys():
      for val in self.sym_table[name]:
        self.wl.append(val)
    elif name in self.globals_map.keys():
      for val in self.globals_map[name]:
        self.wl.append(val)
    else:
      self.result.append(name)

  def visit_Call(self, node):
    for attr in node.__dict__.keys():
      # skip value in function call position. Will be processed!
      if node.__dict__[attr] != node.func: 
        if ins(node.__dict__[attr],ast.AST):
          super(ClosureAnalyzer, self).visit(node.__dict__[attr]);
        elif ins(node.__dict__[attr],list):
          for elt in node.__dict__[attr]:
             super(ClosureAnalyzer, self).visit(elt);            

class TestAnalyzer(ast.NodeVisitor):
 
  def __init__(self):
    self.sym_table_analyzer = None
    self.globals_map = crawler.get_function_map()[3]
          
  def visit_FunctionDef(self, node):
    if node.name != 'fit': return 
    self.sym_table_analyzer = SymTableAnalyzer()
    self.sym_table_analyzer.visit(node);
    for key in self.sym_table_analyzer.sym_table.keys():
         print("A key in sym table: ",key);
         for val in self.sym_table_analyzer.sym_table[key]:
            print("-- A val: ",ast.unparse(val))
    super(TestAnalyzer, self).generic_visit(node);

  def visit_Call(self, node):
    if self.sym_table_analyzer == None: return
    closure_analyzer = ClosureAnalyzer(self.sym_table_analyzer.sym_table,node.func,self.globals_map)
    print("\nAnd the call is: "+ast.unparse(node));
    closure_analyzer.solve_wl()
    for val in closure_analyzer.result:
      print("-- A val: ",val)
    super(TestAnalyzer, self).generic_visit(node);

def main(operator_main):

  #Analyzer takes main class, e.g., PCA, and main method, e.g., fit and constructs a call graph starting at main method.

   analyzer = TestAnalyzer()
   try: 
     with open("/PATH_TO/PCA.py", "r") as source:
       tree = ast.parse(source.read(), type_comments=True, feature_version=sys.version_info[1])
       analyzer.visit(tree)
   except SyntaxError:
     print("Oops, Syntax error: ")

if __name__ == "__main__":

  main(sys.argv[1])
  print("DONE")


    
