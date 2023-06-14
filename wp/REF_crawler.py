import ast
import sys
import os

import config

def ins(obj,cls):
  return isinstance(obj,cls);

function_map = {}
bases_map = {}
imports_map = {}
globals_map = {}

imported_types_map = {}
imported_functions_map = {}

BAD_MODULES = ['pyarrow','cupy','modin','datatable','cudf','dask','sklearn.cross_validation','sparse']

def _import_type_map(import_node,imported_modules_list,encl_module):

  if encl_module not in imported_functions_map.keys(): imported_functions_map[encl_module] = []
  if encl_module not in imported_types_map.keys(): imported_types_map[encl_module] = []
 
  for imported_module in imported_modules_list:
    try:
      code = ast.unparse(import_node)+"; global val; val="+imported_module+'.__all__'
      # print("!!! And the source segment: ",code);
      exec(code);
      # print("And the result is: ",val);
      for the_type in val:
        code2 = ast.unparse(import_node)+"; global val2; val2 = type("+imported_module+'.'+the_type+')'
        exec(code2)
        if 'function' in str(val2):
          imported_functions_map[encl_module].append(imported_module+'.'+the_type)
        elif 'type' in str(val2):
          imported_types_map[encl_module].append(imported_module+'.'+the_type)

    except ModuleNotFoundError as e:
      print("ModuleNotFoundError", e)
    except AttributeError as e:
      print("AttributeError", e);

def print_map(the_map):
  print("\n\n Printing the map\n")
  for key in the_map.keys():
    print("A key: ", key);
    for val in the_map[key]:
      print("---- A value: ",val)
  print("Size of the map: ",len(the_map.keys()))

def print_simple_map(the_map):
  print("\n\n Printing the simple map, keys only:")
  for key in the_map.keys():
    print("A key: ", key)

class Crawler(ast.NodeVisitor):
    def __init__(self,file_name,package_name):
      # A Class.func_name map
      self.func_stack = ['None']
      self.class_stack = ['None']
      self.module = file_name
      self.package_name = package_name
      imports_map[self.module] = ""

    #TODO: Have to handle aliased classes and functions
    def visit_Import(self, node):
      if self.package_name in ast.unparse(node): return #TODO: Revisit
      for bad in BAD_MODULES:
        if bad in ast.unparse(node): return #pyarrow doesn't import on my machine...
      #print("Visiting import: ",ast.unparse(node)," and ast dump: ",ast.dump(node))
      imports = imports_map[self.module]
      imports_map[self.module] = imports+ast.unparse(node)+";"
      # print("New imports map: ", imports_map[self.module]) 
      imported_modules_list = [] 
      for alias in node.names:
        if alias.asname:
          imported_modules_list.append(alias.asname)
        else:
          imported_modules_list.append(alias.name)
        # print("FOUND an alias node: ", ast.dump(alias))
      #_import_type_map(node,imported_modules_list,self.module)


    #TODO: Have to handle aliased classes and functions
    def visit_ImportFrom(self, node):
      if self.package_name in ast.unparse(node): return
      for bad in BAD_MODULES:
        if bad in ast.unparse(node): return
      #print("Visiting importFrom: ",ast.unparse(node)," and ast dump: ",ast.dump(node))
      if node.level == 0:
        imports = imports_map[self.module]
        imports_map[self.module] = imports+ast.unparse(node)+";"
        #print("New imports map: ", imports_map[self.module])
        #_import_type_map(node,node.module)
      # for alias in node.names:
      #  if alias.asname:
          # print("FOUND an alias node: ", ast.dump(alias))

    def visit_ClassDef(self, node):
      self.class_stack.append(node.name);
      bases_list = [];
      for base in node.bases:
        if ins(base,ast.Name):
          bases_list.append(base.id);
        elif ins(base,ast.Attribute): #TODO: Revisit
          bases_list.append(base.attr)
          if config.PRINT_DEBUG: print("HERE I AM: Attribute base: ",ast.unparse(base))
      bases_map[self.module+":"+node.name] = bases_list;
      super(Crawler, self).generic_visit(node);
      self.class_stack.pop();

    def visit_FunctionDef(self, node):
      # TODO: We ignore nested function definitions for now!
      if self.func_stack[-1] == 'None':
        func_name = self.module+":"+self.class_stack[-1]+":"+node.name
      else:
        # Inner function
        func_name = self.module+":"+'None'+":"+self.func_stack[-1]+';'+node.name
        #print(func_name)
        #assert False
      if func_name in function_map.keys():
        if config.PRINT_DEBUG: print("WARNING: redefinition: ", func_name)  # TODO: resolve redefinition
      function_map[func_name] = node
      self.func_stack.append(node.name);
      super(Crawler, self).generic_visit(node);
      self.func_stack.pop(); 

    def visit_AugAssign(self, node):
     # This is target += value
     # TODO: Needs testing!
     # print("\nAugAssign: ",ast.dump(node, include_attributes=True));
     if self.class_stack[-1] != 'None' or self.func_stack[-1] != 'None': return
     self._Assign_Helper(node.target,ast.BinOp(left=node.target,right=node.value,op=node.op));

    def visit_AnnAssign(self, node):
      # print("\nAnnAssign: ",ast.dump(node, include_attributes=True));
      if self.class_stack[-1] != 'None' or self.func_stack[-1] != 'None': return 
      if node.value != None:
        self._Assign_Helper(node.target,node.value);

    def visit_Assign(self, node):
      # targets, value; can have tuple, e.g., x,y = f() and multiple targets x = y = f()                                                                  
      # print("\nAssign: ",ast.dump(node));                                                                              
      # self.visit(node.value);                                                                                                                      
      # print("Here in Assign ",ast.dump(node))                                                                                                   
      if self.class_stack[-1] != 'None' or self.func_stack[-1] != 'None': return
      for target in node.targets:
        if ins(target,ast.Tuple):
          sub = 0;
          for elem in target.elts:
            new_value = ast.Subscript(value=node.value,slice=ast.Index(value=ast.Constant(value=sub)));
            self._Assign_Helper(elem,new_value);
            sub+=1;
        else:
          self._Assign_Helper(target,node.value);

    def _Assign_Helper(self, lhs, rhs):
      # store in symbol table only if lhs is a Name                                                                                                           
      # print("Here in Assign Helepr recording lhs ",ast.dump(lhs))                                                                                         
      if ins(lhs,ast.Name) and "__" not in lhs.id:
        # if self.module in globals_map.keys():
        if lhs.id in globals_map.keys():
          globals_map[lhs.id].append(rhs)
        else:
          globals_map[lhs.id] = [rhs]
  
# Top-level function. Crawls through the sklearn directory and creates a map: 
# Full_file_path:Class_name:Function_name -> Function_Def AST node
# If function is not part of a class then Class_name is None 
def get_function_map(package_dir, package_name):  
  global function_map
  global bases_map
  global imports_map
  global globals_map

  function_map = {}
  bases_map = {}
  imports_map = {}
  globals_map = {}

  def inSkiplist(file_name):
    skiplist = []
    for s in skiplist:
      if s in file_name:
        return True

    # numpy. Skip testing files
    if "/Lib/site-packages/numpy" in file_name:
      if "test_" in file_name:
        return True

    return False

  for path, directories, files in os.walk(package_dir):  
    
    # To skip "tests" directory in sklearn
    if "tests" in path: continue
    
    for file in files:
      if file.endswith(".py"):
        file_name = os.path.join(path, file)
        if config.PRINT_DEBUG: print("Analyzing: ",file_name)
        if inSkiplist(file_name):
          continue

        crawler = Crawler(file_name,package_name)
        try: 
          with open(file_name, "r") as source:
            tree = ast.parse(source.read(), type_comments=True, feature_version=sys.version_info[1])
            crawler.visit(tree)
        except SyntaxError:
          print("Oops, Syntax error: ")
          assert False, "REF_crawler.py SyntaxError"

  #print_simple_map(function_map)
  #print_map(bases_map);
  return function_map, bases_map, imports_map, globals_map 

if __name__ == "__main__":
  #get_function_map()
  print("> REF_crawler.py: NOTHING IS HERE")
