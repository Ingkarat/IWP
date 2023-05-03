import ast
import sys
import os

def ins(obj,cls):
  return isinstance(obj,cls);

function_map = {}
bases_map = {}

def print_map():
  print("\n\n Printing the funciton map\n")
  for key in function_map.keys():
    print("A key: ", key);
  print("Size of function_map: ",len(function_map.keys()))

def print_bases():
  print("\n\n Printing the funciton map\n")
  for key in bases_map.keys():
    print("A key: ", key, bases_map[key]);
  print("Size of bases_map: ",len(bases_map.keys()))

class Crawler(ast.NodeVisitor):
    def __init__(self,file_name):
      # A Class.func_name map
      self.class_stack = ['None']
      self.module = file_name
      self.func_stack = ['None']

    def visit_ClassDef(self, node):
      self.class_stack.append(node.name);
      bases_list = [];
      for base in node.bases:
        if ins(base,ast.Name):
          bases_list.append(base.id);
      bases_map[self.module+":"+node.name] = bases_list;
      super(Crawler, self).generic_visit(node);
      self.class_stack.pop();

    def visit_FunctionDef(self, node):
      # TODO: We ignore nested function definitions for now!
      #func_name = self.module+":"+self.class_stack[-1]+":"+node.name

      if self.func_stack[-1] == 'None':
        func_name = self.module+":"+self.class_stack[-1]+":"+node.name
      else:
        # Inner function
        func_name = self.module+":"+'None'+":"+self.func_stack[-1]+';'+node.name
        #print(func_name)
        #assert False

      if 0:
        if func_name in function_map.keys():
          print("WARNING: redefinition: ", func_name)
      function_map[func_name] = node

      self.func_stack.append(node.name);
      super(Crawler, self).generic_visit(node);
      self.func_stack.pop(); 
      
      #print(type(node.args))
      #if node.args.vararg is not None:
      #  print("\n",func_name)
      #  print(ast.dump(node.args))
      
# Top-level function. Crawls through the sklearn directory and creates a map: 
# Full_file_path:Class_name:Function_name -> Function_Def AST node
# If function is not part of a class then Class_name is None 
def get_function_map():      

  sklearn_dir = "C:/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/sklearn"
  pytorch_dir = "C:/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/torch"
  sklearn_pandas_dir = "C:/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/sklearn_pandas"
  mapie_dir = "C:/Users/.../AppData/Local/Programs/Python/Python39/Lib/site-packages/mapie"
  analyzing_dir = sklearn_dir

  def inSkiplist(file_name):
    #skiplist = ["torch/_tensor_docs.py", "torch/_torch_docs.py"]
    #skiplist = ["numpy/polynomial/_polybase.py", "numpy/testing/"]
    #skiplist = ["tensorflow/python/framework/config.py"]
    skiplist = ["sklearn/feature_extraction/tests/test_text.py", "sklearn/preprocessing/tests/test_encoders.py"]
    for s in skiplist:
      if s in file_name:
        return True
    
    # numpy
    if "/Lib/site-packages/numpy" in file_name:
      if "test_" in file_name:
        return True

    return False

  for path, directories, files in os.walk(analyzing_dir):  
  # for file in os.listdir(sklearn_dir):
    #if "tests" in path: continue
    for file in files:
      if file.endswith(".py"):
        file_name = os.path.join(path, file).replace("\\","/").replace("C:","") # remove "C:" to avoid split() issue
        print("Analyzing: ",file_name);
        #if "_test.py" in file_name:
        #  continue
        if inSkiplist(file_name):
          continue

        crawler = Crawler(file_name)
        try: 
          with open(file_name, "r") as source:
            tree = ast.parse(source.read(), type_comments=True, feature_version=sys.version_info[1])
            crawler.visit(tree)
        except SyntaxError:
          print("Oops, Syntax error?")
          assert False, "crawler.py SyntaxError"

  #print_map();
  #print_bases()
  print("\n=== END OF Crawler.py ===\n\n")
  return function_map, bases_map 

if __name__ == "__main__":
  get_function_map()
  #print_map()
  #print(bases_map)
  print("DONE")