#!/usr/local/bin/python3.9

import copy
import ast
import sys
import os

import REF_graph
import REF_crawler
import REF_symbol_table

def ins(obj,cls):
  return isinstance(obj,cls);

class CallGraphAnalyzer(ast.NodeVisitor):
  
  def __init__(self,function_map,bases_map,imports_map,globals_map):
    
    self.call_graph = REF_graph.Graph();
    self.visited = [];
    self.curr_func = ""
    self.curr_symbol_table = {}
    self.worklist = []
    self.function_map = function_map
    self.bases_map = bases_map
    self.imports_map = imports_map
    self.globals_map = globals_map

    self.num_calls = 0
    self.num_unresolved = 0
    self.unresolved = []
    self.num_libcalls = 0
    self.libcalls = [] 
 
  def visit_FunctionDef(self,node):
    return 

  def visit_Call(self,node):
    # Call(expr func, expr* args, keyword* keywords)
    print("\n\nVisiting call: ", ast.unparse(node),ast.dump(node))
    found = False
    self.num_calls += 1
    if ins(node.func,ast.Name):
      print("Here, trying a Name_call")
      found = self.resolve_Name_call(node,node)
    elif ins(node.func,ast.Attribute): 
      if ins(node.func.value,ast.Name) and self.resolve_base(node.func.value.id) and len(node.args)>0 and ins(node.args[0],ast.Name) and node.args[0].id == 'self':
        print("Here, trying a ClassName_Method_call")
        found = self.resolve_ClassName_Method_call(node,node)
      elif ins(node.func.value,ast.Name) and node.func.value.id == 'self': 
        print("Here is a call through self: ",ast.unparse(node))
        print("Call is in curr_func: ", self.curr_func, self.curr_class)
        found = self.resolve_Self_Method_call(node)
      elif ins(node.func.value,ast.Call) and ins(node.func.value.func,ast.Name) and node.func.value.func.id=='super':
        print("Here I am call through super?", ast.unparse(node))
        found = self.resolve_Super_Method_call(node,node)
      else:
        print("Here I am in an Attribute call. Will resolve if prefix is a packge and I find the name there...")
        package = ast.unparse(node.func.value)
        new_node = copy.deepcopy(node)
        new_node.func = ast.Name(id=node.func.attr)
        found = self.resolve_Name_call(new_node,node,package)      
        if found:
          print("FOUND! ",ast.unparse(node))

    # print("Here I am now...: ",ast.dump(node),found)
    found_before = found
    found = found or interpret_node_natively(node,self.imports_map,self.curr_func,self.globals_map)
    if not found_before and found: 
      self.num_libcalls=self.num_libcalls+1;
      self.libcalls.append(self.curr_func+":"+ast.unparse(node))

    #if not found:
      # Now trying to resolve calls y.m()
      
    found = found or self.process_local_analysis(node)

    if not found:
      self.num_unresolved += 1
      self.unresolved.append(self.curr_func+":"+ast.unparse(node))
      print("UNRESOLVED Call: ", ast.unparse(node))
      print("Function call in ",self.curr_func," is something else...\n")
    super(CallGraphAnalyzer, self).generic_visit(node);    

  def process_local_analysis(self,node):  
    assert ins(node,ast.Call)
    found = False
    #print("Here processing yet Unresolved Call: ", ast.unparse(node))
    if ins(node.func,ast.Attribute): 
      if ins(node.func.value,ast.Name):
        #print("\nHERE: Function call in ",self.curr_func," is some other kind of call...")
        #print("And the call is: "+ast.unparse(node));
        closure_analyzer = REF_symbol_table.ClosureAnalyzer(self.curr_symbol_table,node.func.value,self.globals_map,self.bases_map)
        closure_analyzer.solve_wl()
        for val in closure_analyzer.result:
          #print("-- A val: ",val)
          new_node = copy.deepcopy(node)
          if ins(val,str):         
            new_node.func.value = ast.Name(id=val)
            #print("-- And the new node is: ",ast.unparse(new_node),new_node.func)
            found = self.resolve_ClassName_Method_call(new_node,node) or found
          else:
            new_node.func.value = val
            #print("-- And the new node is: ",ast.unparse(new_node),new_node.func)
            found = found or interpret_node_natively(new_node,self.imports_map,self.curr_func,self.globals_map)
      if not found:
        pass
        '''
        list_node = ast.Call(func=ast.Attribute(value=ast.List(elts=[ast.Constant(value=0)]), attr=node.func.attr), args=[], keywords=[])
        print("Here trying that new great idea, trying list:...",ast.unparse(list_node))
        found = interpret_node_natively(list_node,self.imports_map,self.curr_func,self.globals_map)
        print("And found is: ",found)
        dict_node = ast.Call(func=ast.Attribute(value=ast.Dict(keys=[ast.Constant(value=0)],values=[ast.Constant(value=1)]), attr=node.func.attr), args=[], keywords=[])
        found = found or interpret_node_natively(dict_node,self.imports_map,self.curr_func,self.globals_map)
        print("And found is: ",found)
        str_node = ast.Call(func=ast.Attribute(value=ast.Constant(value='ana'), attr=node.func.attr), args=[], keywords=[])
        found = found or interpret_node_natively(str_node,self.imports_map,self.curr_func,self.globals_map)          
        print("And found is: ",found)
        '''

      # TODO: Attr call. Do local analysis of receiver and then call interpret natively, then 
      # There are 3 cases: 
      # 1) a native or out-of-scope module (CPython) call, 
      # 2) a receiver call to a likely in-scope method that can be locally resolved (or not), and 
      # 3) a standard library call, e.g., rec_expr.attr and attr points to some method in standard library 
      # print("HERE: Attribute call in ",self.curr_func," is attribute call... TO BE HANDLED LATER \n")
      # print("And the call is: "+ast.unparse(node))
    else:
      # print("HERE 2: Function call in ",self.curr_func," is some other kind of call...")
      closure_analyzer = REF_symbol_table.ClosureAnalyzer(self.curr_symbol_table,node.func,self.globals_map,self.bases_map)
      # print("And the call is: "+ast.unparse(node));
      closure_analyzer.solve_wl()
      for val in closure_analyzer.result:
        # print("-- A val: ",val)
        new_node = copy.deepcopy(node)
        if ins(val,str):
          if 'self.' not in val:
            new_node.func = ast.Name(id=val)
            #print("-- And the new node is: ",ast.unparse(new_node),new_node.func)
            found = self.resolve_Name_call(new_node,node) or found
        else:
          new_node.func = val
          found = found or interpret_node_natively(new_node,self.imports_map,self.curr_func,self.globals_map) 
      if ins(node.func,ast.Name):
        print("Trying inner now:")
        encl_func_name = self.curr_func.split(':')[2]
        print("Enclosing method:",encl_func_name)
        func_list = self.resolve_name(encl_func_name+';'+node.func.id)
        if func_list:
          for func_name in func_list:
            print("Call: ", ast.unparse(node), " in ", self.curr_func)
            print('INNER FUNC IN NAME Func call from ',self.curr_func," to ",func_name);
            self.new_edge(self.curr_func,func_name,node,None)
            found = True  
    return found

  def new_edge(self,caller,callee,node,curr_class): 
    print("Added a new call graph edge: ", caller, " to ", callee, " in ", curr_class)
    self.call_graph.addEdge(REF_graph.Edge(caller,callee,node)) #TODO: Here add rec class to edge label?
    if (curr_class,callee) not in self.visited:
      self.visited.append((curr_class,callee))
      self.worklist.append((curr_class,callee))

  def solve_worklist(self,main_class,main_func):
    return #PANIC, REMOVE THIS
    cl = self.resolve_base(main_class)
    print('cl is',cl)
    if cl == None:
      raise ValueError("Check the operator, cannot find opearator class ",main_class)
    print('main_class: ',main_class, 'and main_func: ',main_func)
    func = self.resolve_self(main_func,cl)
    print('func is',func)
    if func == None:
      raise ValueError("Check the operator, cannot find fit method for ",main_class)
    
    if (cl,func) not in self.visited: 
      self.visited.append((cl,func))
      self.worklist.append((cl,func))
    while not self.worklist == []:
      (class_name,func_name) = self.worklist.pop(0)
      print("Just popped: ",(class_name,func_name))
      func_ast = self.function_map[func_name]
      self.curr_class = class_name
      self.curr_func = func_name
      sym_table_analyzer = REF_symbol_table.SymTableAnalyzer()
      sym_table_analyzer.visit(func_ast)
      self.curr_symbol_table = sym_table_analyzer.sym_table
      super(CallGraphAnalyzer, self).generic_visit(func_ast)
      
  # requires: ins(node,ast.Call) and ins(node.func,ast.Name)
  # returns: Boolean, True if resolved, False otherwise 
  def resolve_Name_call(self,node,orig_ast_node,package=None):
    found = False
    print("Function is a Name_call.",node.func.id)                                                                                                           
    func_list = self.resolve_name(node.func.id,package)
    if func_list:
      for func_name in func_list:
        # print("Call: ", ast.dump(node))                                                                                                         
        print('IN NAME Func call from ',self.curr_func," to ",func_name);                                                                              
        self.new_edge(self.curr_func,func_name,orig_ast_node,None)
        found = True
    # TODO: Add handling of constructor calls                                                                                                    
    elif self.resolve_base(node.func.id):
      print("Constructor call: ", ast.unparse(node))
      full_class_name = self.resolve_base(node.func.id)
      if full_class_name: 
        found = True # It's resolved even if there's no __init__ 
        func_init = self.function_map.get(full_class_name+':__init__',None)
        if func_init:
          # print("HERE2...",full_class_name)
          self.new_edge(self.curr_func,full_class_name+':__init__',orig_ast_node,full_class_name)
    elif node.func.id == 'super':
       print("Constructor call through super()", ast.unparse(node))
       new_node = copy.deepcopy(node)
       new_node.func = ast.Attribute(value=ast.Call(func=ast.Name(id='super'),args=[],keywords=[]),attr='__init__')
       print("And the new node is: ",ast.unparse(new_node))
       found = self.resolve_Super_Method_call(new_node,node)
    return found

  # requires: the_name:String is an unqualified non-member function name. 
  # returns: return:Option[[str]] a list of expanded non-member function names 
  def resolve_name(self,the_name,package=None):
    func_list = []
    for full_func_name in self.function_map.keys():
      full_func_name_split = full_func_name.split(':')
      if full_func_name_split[1] == 'None' and full_func_name_split[2] == the_name:
        if package == None or (package and package in full_func_name):
          func_list.append(full_func_name)
    if func_list == []:
      return None
    else:
      return func_list

  # requires: call is ClassName.MethodName(self)
  def resolve_ClassName_Method_call(self,node,orig_node):
    found = False
    #print("Call: ", ast.dump(node))                                                                                                                          
    #print("Function call is DIRECT call ClassName.methodname(self)")
    base_class = self.resolve_base(node.func.value.id)
    if base_class:                                                                                     
      func_name = self.resolve_self(node.func.attr,base_class)
      # print("func_name",func_name)                                                                                                                            
      if func_name in self.function_map.keys():
        self.new_edge(self.curr_func,func_name,orig_node,self.resolve_base(node.func.value.id)) #TODO: CHECK THIS OUT!!!                                     
        found = True
    return found      

  def resolve_Self_Method_call(self,node):
    #print("Here is a call through self: self.func_name()")
    if self.curr_class == None: #TODO: Fix the mix of 'None' and None!
      print("WARNING: Could not resolve call through self:", ast.unparse(node), ' in function: ',self.curr_func)
      return False 
    found = False   
    func_name = self.resolve_self(node.func.attr,self.curr_class)
    if func_name in self.function_map.keys():
      self.new_edge(self.curr_func,func_name,node,self.curr_class);
      found = True
    return found

  def resolve_Super_Method_call(self,node,orig_node):
    found = False
    curr_func_split = self.curr_func.split(':')
    encl_class = curr_func_split[0]+":"+curr_func_split[1]
    bases = self.bases_map[encl_class]
    print("Bases: ",bases)
    for base in reversed(bases):
      full_base = self.resolve_base(base)
      print("base and full_base", base, full_base)
      if not full_base: continue #It is possible that Base is NOT in scope, move on to previous. THIS IS UNSOUND.
      func_name = self.resolve_self(node.func.attr,full_base);
      if not (func_name == None):
        self.new_edge(self.curr_func,func_name,orig_node,self.curr_class)
        found = True
        break;
    return found 

  # requires: method name and fully qualified class name (receiver class) str and str
  # returns: Option[str]
  def resolve_self(self,the_name,curr_class):
    print("the_name:", the_name, "curr_class: ",curr_class)
    # TODO: Check Semantics of Inheritance!!!
    full_name = curr_class+":"+the_name
    print('full_name:', full_name)
    if full_name in self.function_map.keys():
      return full_name
    else:
      bases = self.bases_map[curr_class]
      print(">bases: ",bases)
      # for pandas
      if 1:
        ccc = curr_class.split(":")[-1]
        safe_list = ["BlockManager"]
        if ccc in bases:
          if ccc not in safe_list:
            #print("...\n",self.function_map.keys(),"...\n")
            print(">bases: ",bases, "ccc: ", ccc)
            # tensorflow, Ugly hack, TODO
            if "/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/tensorflow\\python\\framework\\dtypes.py:DType:" in full_name:
              return full_name
            if full_name == "/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/tensorflow\\python\\keras\\initializers\\initializers_v1.py:RandomNormal:items":
              return full_name
            assert False, "PANIC"
          bases.remove(ccc)

      for base in reversed(bases):
        full_base = self.resolve_base(base)
        #print("base: ",base)
        if full_base:
          full_name = self.resolve_self(the_name,full_base)
          if full_name: 
            return full_name
      #print("NONE!!!!")
      return None
    
  # takes the base (class) name, unqualified
  # returns: Option[str] fully qualified class name 
  def resolve_base(self,the_base_name):
    for base in self.bases_map.keys():
      #print(base.split(':')[2], the_base_name)
      #print(base)
      if base.split(':')[1] == the_base_name: 
      # TODO: what if there are more than one classes named Base???
        return base
    return None

def interpret_node_natively(node,imports_map,curr_func,globals_map):
  print("Before interpret: ", ast.unparse(node))
  resolved = False
  if ins(node.func,ast.Name) and node.func.id == "super": return resolved
  try:
      call_node = copy.deepcopy(node)
      call_node.args = []
      call_node.keywords = []
      code = ast.unparse(call_node)
      print("The source segment: ",imports_map[curr_func.split(':')[0]]+" "+code);
      exec(imports_map[curr_func.split(':')[0]]+" global val; val="+code)
  except NameError:
      print("A name error for ", ast.dump(call_node)," and code ",code," error is ")
      #Let's try evaluating the rhs if it's a name...
      pass
  except:
      print("Unexpected error:", "0: ",sys.exc_info()[0],"1: ",str(sys.exc_info()[1]),code)
      if 'argument' in str(sys.exc_info()[1]) or 'arg' in str(sys.exc_info()[1]) or 'operand' in str(sys.exc_info()[1]): 
        resolved = True
  else:
      print("EVALUATED JUST FINE", ast.dump(call_node)," and code ", code," and val = ",val)
      resolved = True
  return resolved


def main(operator_main):

   # doing nothing here
   ...

if __name__ == "__main__":

  main(sys.argv[1])
  print("DONE")


    
