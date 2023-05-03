#!/usr/local/bin/python3.8

import ast

# Top level Raise_Analyzer.visit call should be made on a FunctionDef node
class Raise_Analyzer(ast.NodeVisitor):
    def __init__(self):
      self.raise_nodes = []
      self.top_level_func = True;

    def IsThisInterestedSingleRaise(self, node):
      print("\nRaise: ",ast.dump(node))
      #InterestingRaise = "A sparse matrix was passed, but dense"
      #InterestingRaise = "Training only accepts Dataset object 1"
      InterestingRaise = "Pandas DataFrame with mixed "
      #InterestingRaise = "num_boost_round should be greater than zero."

      if isinstance(node.exc,ast.Call):
        #print(node.exc.args[0].func.value.value)
        if 0:
          if isinstance(node.exc.args[0], ast.JoinedStr):
            if node.exc.args[0].values[0].value == "X has ":
             return True
        if 1:
          if len(node.exc.args) == 0:
            return False
          if isinstance(node.exc.args[0], ast.Constant):
            if InterestingRaise in node.exc.args[0].value:
              return True
          elif isinstance(node.exc.args[0], ast.BinOp):
            #print(">>> ", ast.dump(node.exc.args[0]))
            if isinstance(node.exc.args[0].left, ast.Constant):
              if InterestingRaise in node.exc.args[0].left.value:
                return True
      return False

    def visit_FunctionDef(self, node):
      #print("\nFunctionDef: ",ast.dump(node));
      if self.top_level_func == False: return;
      self.top_level_func = False;
      super(Raise_Analyzer, self).generic_visit(node);

    def visit_Raise(self, node):
      #print("\nRaise: ",ast.dump(node, include_attributes=True));
      lookAtSpecificRaise = False
      if lookAtSpecificRaise:
        if self.IsThisInterestedSingleRaise(node):
          self.raise_nodes.append(node);
      else:
        self.raise_nodes.append(node)


