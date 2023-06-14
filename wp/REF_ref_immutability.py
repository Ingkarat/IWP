#!/usr/local/bin/python3.9

import ast
import sys
import os

import config
import REF_crawler
import REF_call_graph

def ins(obj,cls):
  return isinstance(obj,cls);

class Constraint:
  def __init__(self,lhs,rhs,kind):
    self.lhs = lhs
    self.rhs = rhs
    self.kind = kind
  def solve(self):
    pass
  def print_side(self,side):
    if isinstance(side,str):
     return side
    elif ins(side,tuple):
     return '('+self.print_side(side[0])+' |> '+side[1]+')'
    else:
     return ast.unparse(side)
  def print_constraint(self):
    print(self.print_side(self.lhs), self.kind, self.print_side(self.rhs))
      
class RefImmutabilityAnalyzer(ast.NodeVisitor):
  
  def __init__(self, call_graph, unresolved, libcalls, function_map): 
    self.constraints = []
    self.constraints_hash = []
    self.call_graph = call_graph
    self.unresolved = unresolved
    self.libcalls = libcalls
    self.function_map = function_map
    self.curr_func = ""
    self.result_stack = []
    self.worklist = []
    
    self.self_locs = {} # str (:self) -> [fields]

    self.node_to_constraints = {} # str|Node -> [Constraint]
    self.attr_to_constraints = {} # str -> [Constraint]
    self.node_to_type = {} # str|Node -> mutable|poly|readonly
    self.attr_to_type = {} # str -> mutable|poly|readonly

    self.unresolved_whitelist = ["apply","astype","to_coo","issparse","any","all","asformat","tocsr","getformat","startswith","endswith","copy"] # treated as LIBCALLS while rest of unresolved are treated as UNRESOLVED.
    # LIBCALLS have type poly->poly, UNRESOLVED have types mutable->?
    self.unresolved_blacklist = ["append","update","remove","sort"]   

  def visit_AugAssign(self, node):
    # This is target += value 
    # TODO: Needs testing!
    # print("\nAugAssign: ",ast.unparse(node)) #, include_attributes=True)); 
    self._Assign_Helper_Wrap(node.target,ast.BinOp(left=node.target,right=node.value,op=node.op));
    
  def visit_AnnAssign(self, node):
    # print("\nAnnAssign: ",ast.unparse(node)) #, include_attributes=True)); 
    if node.value != None:
      self._Assign_Helper_Wrap(node.target,node.value);

  def visit_Assign(self, node):
    # targets, value; can have tuple, e.g., x,y = f() and multiple targets x = y = f()  
    #print("\nAssign: ",ast.dump(node));
    #print("\nAssign ",ast.unparse(node))
    for target in node.targets:
      self._Assign_Helper_Wrap(target,node.value);
  
  def _Assign_Helper_Wrap(self,target,rhs):
    if ins(target,ast.Tuple):
      elem_rhs = ast.Subscript(value=rhs,slice=ast.Constant(value=1))
      for elem_target in target.elts:
        self._Assign_Helper(elem_target,elem_rhs)
    else:
      self._Assign_Helper(target,rhs)

  def _Assign_Helper(self, lhs, rhs):
    # store in symbol table only if lhs is a Name or an Attribute self.attr
    if config.PRINT_DEBUG: 
      print("\nHere in Assign Helper, lhs = rhs", ast.unparse(lhs)," = ",ast.unparse(rhs))
      print("\nHere in Assign Helper, recording lhs:",ast.unparse(lhs),"!",ast.dump(lhs))
      print("\nHere in Assign Helper, recording rhs:",ast.unparse(rhs),ast.dump(rhs))
    self.result_stack.clear()
    self.visit(lhs)
    lh_sides = []
    while len(self.result_stack) > 0:
      lh_sides.append(self.result_stack.pop())
    self.visit(rhs)
    rh_sides = []
    while len(self.result_stack) > 0:
      rh_sides.append(self.result_stack.pop())
    # print("lh_sides: ",lh_sides)
    # print("rh_sides: ",rh_sides)
    
    for the_lhs in lh_sides:
      if (ins(the_lhs,ast.Attribute) and not (ins(the_lhs.value,ast.Name) and the_lhs.value.id=="self")) or ins(the_lhs,ast.Subscript):
        if ins(the_lhs.value,ast.Name):
          the_lhs_value = self.curr_func+":"+ast.unparse(the_lhs.value)
        else:
          the_lhs_value = the_lhs.value

        if not (ins(the_lhs_value,str) and 'check_array:dtypes_orig' in the_lhs_value):  
          # WARNING: prevents a single source of substantial imprecision in sklearn's check_array!
          # ingoring this mutation is safe because dtypes_orig = list(array.dtypes) which copies the immutable values of dtypes arrays
          self._add_Constraint(the_lhs_value,"mutable","SUB")
        #print("Mutable constraint")
      if ins(the_lhs,ast.Attribute) and ins(the_lhs.value,ast.Name) and the_lhs.value.id=="self":
        self._add_to_map(self.self_locs,self.curr_func+":self",the_lhs.attr)    
      for the_rhs in rh_sides:
        self._add_Constraint(the_rhs,the_lhs,"SUB")
        
    #else:
    # print("BAD ASSIGN: ",ast.unparse(lhs)," = ",ast.unparse(rhs))
 
  def visit_Name(self, node):
    var_name = self.curr_func+":"+ast.unparse(node)
    #print("Visiting Name: "+self.curr_func+":"+ast.unparse(node));
    self.result_stack.append(var_name)

  def _add_to_map(self,map,key,value):
    if key in map.keys():
      if not value in map[key]:
        map[key].append(value)
    else:
      map[key] = [value]

  def _in_map(self,map,key,value):
    if not key in map.keys():
      return False
    else:
      if value in map[key]:
        return True
      else:
        return False

  def _record_side(self, side, cons):
    if ins(side,tuple):
      the_node = side[0]
      adaptee = side[1]
      assert ins(adaptee,str)
      if ":" in adaptee: # i.e., adaptee is a ret_var or param, not a attr/field
        self._add_to_map(self.node_to_constraints,adaptee,cons)
        self.node_to_type[adaptee] = 'readonly'
      else:
        self._add_to_map(self.attr_to_constraints,adaptee,cons)
        self.attr_to_type[adaptee] = 'readonly'
    elif side == 'mutable':
      the_node = None
    else:
      the_node = side
    if the_node:
      self.node_to_type[the_node] = 'readonly'
      self._add_to_map(self.node_to_constraints,the_node,cons)

  def _add_Constraint(self, lhs, rhs, kind):
    # TODO: Have to check if constraint already in list. Possibly change rep of constraints
    cons = Constraint(lhs,rhs,kind)
    cons_hash = str(lhs)+str(rhs)+kind
    if cons_hash in self.constraints_hash:
      return
    self.constraints_hash.append(cons_hash)

    self._record_side(lhs,cons)
    self._record_side(rhs,cons)
    
    #cons.print_constraint()
    self.constraints.append(cons) 
    if rhs == 'mutable' and not (cons in self.worklist):
      self.worklist.append(cons)
    if ins(rhs,tuple) and self._is_self(rhs[1]) and not (cons in self.worklist): 
      self.worklist.append(cons)

  def visit_Attribute(self, node):
    self.result_stack.append("MARKER")
    self.visit(node.value)
    #print("Visiting Attribute: "+ast.unparse(node));
    while self.result_stack[-1] != "MARKER":
      value_result = self.result_stack.pop()
      self._add_Constraint((value_result, node.attr),node,"EQU")
    self.result_stack.pop()
    self.result_stack.append(node)    

  def visit_Subscript(self, node):
    #super(RefImmutabilityAnalyzer, self).generic_visit(node);
    #print("Visiting Subscript: "+ast.unparse(node));
    self.result_stack.append("MARKER")
    self.visit(node.value)
    while self.result_stack[-1] != "MARKER":
      value_result = self.result_stack.pop()
      self._add_Constraint((value_result, "[]"),node,"EQU")
    self.result_stack.pop()
    self.result_stack.append(node)

  def visit_Return(self, node):
    #print("Visiting Return node: ",ast.unparse(node)," with dump: ",ast.dump(node))
    self.result_stack.clear()
    if node.value:
      #print("Visiting Return node 2: ",ast.unparse(node)," with dump: ",ast.dump(node))
      self.visit(node.value)
      ret_var_name = self.curr_func+":ret_var"
      while len(self.result_stack) > 0:
        value_result = self.result_stack.pop()
        self._add_Constraint(value_result,ret_var_name,"SUB")
        #print("Visited Return node: ")
        #cons.print_constraint()

  def _Actual_Formal_Assign(self, act, formal, call, callee_kind):
    # actual is an Expression, formal is a full string, call is a Call
    if config.PRINT_DEBUG: 
      print("Actual_Formal_Assign!!!")
      print("act: ", ast.unparse(act), "---> formal: ", formal)
    self.result_stack.append("MARKER")
    self.visit(act)
    lh_sides = []
    while self.result_stack[-1] != "MARKER":
      lh_sides.append(self.result_stack.pop())
    for lhs in lh_sides:
      if callee_kind == 'REGULAR':
        self._add_Constraint(lhs,(call,formal),"SUB")
      elif callee_kind == 'UNRESOLVED':
        self._add_Constraint(lhs,'mutable',"SUB")
      else:
        assert callee_kind == 'LIBCALL'
        self._add_Constraint(lhs,call,"SUB")
    self.result_stack.pop()    

  def _Return_Node(self, node, return_nodes, callee_str, callee_kind):
    if callee_kind == 'REGULAR':
      call_attr = ast.Attribute(value=node,attr=callee_str+':ret_var')
      self._add_Constraint(call_attr,(node,callee_str+':ret_var'),"EQU")
      return_nodes.append(call_attr)
    else:
      assert callee_kind == 'UNRESOLVED' or callee_kind == 'LIBCALL'
      return_nodes.append(node)

  def visit_Call(self, node):

    super(RefImmutabilityAnalyzer, self).generic_visit(node)
    if config.PRINT_DEBUG: 
      print("\nRefImmut. Visiting Call: "+ast.unparse(node))
      print("Call ast dump: "+ast.dump(node))
    actuals = node.args
    dealWithAttribute = False
    if ins(node.func,ast.Attribute): # receiver call value.method(...)                                                                                 
      #assert callee_ast.args.args[0].arg == 'self'
      dealWithAttribute = True
      if config.PRINT_DEBUG: print("NANITF")
      actuals = [node.func.value]+actuals

    return_nodes = []
    if self.curr_func+":"+ast.unparse(node) in self.unresolved:
      if config.PRINT_DEBUG: print("> UNRESOLVED")
      #WARNING: Assumption of unresolved calls having all mutable args is too strong. Treating them like libcalls poly -> poly
      if self._is_in_blacklist(node):
        for act in actuals:
          self._Actual_Formal_Assign(act,None,node,"UNRESOLVED")
        self._Return_Node(node,return_nodes,None,"UNRESOLVED")
      else:
        for act in actuals:
          self._Actual_Formal_Assign(act,None,node,"LIBCALL")
        self._Return_Node(node,return_nodes,None,"LIBCALL")
    elif self.curr_func+":"+ast.unparse(node) in self.libcalls:
      if config.PRINT_DEBUG: print("> LIBCALL")
      for act in actuals:
        self._Actual_Formal_Assign(act,None,node,"LIBCALL")
      self._Return_Node(node,return_nodes,None,"LIBCALL")
    else:
      if config.PRINT_DEBUG: 
        print("> REGULAR")
        print("self.curr_func ",self.curr_func)
      for callee_edge in self.call_graph.getEdgesFromSource(self.curr_func):       
        if callee_edge.label == node: 
          callee_str = callee_edge.tgt
          callee_ast = self.function_map[callee_str]
          if config.PRINT_DEBUG: 
            print("Found a callee: ", callee_str)
            print("The callee_ast: ", ast.dump(callee_ast.args))
          formals = callee_ast.args.posonlyargs+callee_ast.args.args
          if hasattr(callee_ast.args,'vararg') and callee_ast.args.vararg != None:
             diff = len(actuals)-len(formals) #varargs
             for i in range(0,diff):
               dummy_name = self.curr_func+":varargs"
               self._add_Constraint(actuals[len(formals)+i],(dummy_name,"[]"),"SUB")
             actuals = actuals[:len(formals)] # removing the varargs...
             actuals.append(ast.Name(id="varargs"))
             formals.append(callee_ast.args.vararg)

          if ins(node.func,ast.Attribute) and ins(node.func.value,ast.Name) and node.func.value.id == "callback":  
            continue
          if len(formals) == 0 and ins(node.func,ast.Attribute) and (node.func.value.id == "config" or node.func.value.id == "rabit"):
            continue
          if config.PRINT_DEBUG: 
            print("FFF ", formals)
            for ff in formals: print(ast.dump(ff))
            print("AAA ", actuals)
            for aaa in actuals: print(ast.dump(aaa))
          count = 0
          deal = False
          for act in actuals:
            if deal:
              deal = False
              #print("===", ast.dump(act))
              #print(count)
              if ins(act, ast.Name) and act.id == "self":
                continue
            if ins(act, ast.Name):
              if act.id == "self":
                #assert len(actuals) == len(formals) + 1
                containSelf = True
                #assert count == 0
                if count == 0:
                  if formals[count].arg == "self":
                    count = count + 1
                  dealWithAttribute = False
                  continue
            if dealWithAttribute:
              if ins(act, ast.Call):
                if act.func.id == "super" and formals[count].arg == "self":
                  count = count + 1
              else:
                # case Attribute(value=Attribute(value=Name(id='torch', ctx=Load()), attr='autograd'...
                # TODO: Panic this
                #if ins(act, ast.Attribute):
                #  continue


                assert act.id != "self"
                
                # TODO: Panic this
                if len(formals) == 0:
                  continue
                
                if formals[count].arg == "self":
                  count = count + 1
              dealWithAttribute = False
              deal = True
              continue

            if 0:
              print(">> FORMAL ", formals)
              #for fff in formals:
              #  print(ast.dump(fff))
              print("> COUNT ", count)
              print("> ACT ", ast.dump(act))
              #print("> formals[count].arg ", formals[count].arg)
            #TODO: hot fix. change this
            if count == 1 and isinstance(act, ast.Name) and act.id == "data" and len(formals) == 1:
              count = 0
              self._Actual_Formal_Assign(act,callee_str+":"+formals[count].arg,node,"REGULAR")
              count=count+1
              continue
            #TODO: remove this try-except madness and properly fix this
            try:
              self._Actual_Formal_Assign(act,callee_str+":"+formals[count].arg,node,"REGULAR")
            except:
              pass
            count=count+1
            # Skip handling of keywords. keword arguments have default constants. Unlikely (though not impossible!) to have mutable kwarguments.
            # Means we don't have to worry about kwargs.
            #for keyword in node.keywords:
            #  #TODO: Need to handle kwargs here!!! hasattr(keyword,key) == False means kwargs array.
            #  if hasattr(keyword,'value') and keyword.value != None and hasattr(keyword,'arg') and keyword.arg != None:
            #    self._Actual_Formal_Assign(keyword.value,callee_str+":"+keyword.arg,node,"REGULAR")
          #return_nodes.append(ast.Attribute(value=node,attr=callee_str+':ret_var'))
          self._Return_Node(node,return_nodes,callee_str,"REGULAR")
    self.result_stack = self.result_stack + return_nodes



  def visit_For(self,node):
    #print("Visiting FOR ", ast.dump(node))
    #print("Visiting FOR: test ",ast.dump(node.iter)," and target ",ast.dump(node.target))

    self.result_stack.append("MARKER")
    self.visit(node.target)
    targets = []
    while self.result_stack[-1] != "MARKER":
      target_result = self.result_stack.pop()
      targets.append(target_result)
    self.visit(node.iter)
    iters = []
    while self.result_stack[-1] != "MARKER":
      iter_result = self.result_stack.pop()
      iters.append(iter_result)
    self.result_stack.pop()
    for target in targets:
      for iter in iters:
        self._add_Constraint((iter, "[]"),target,"SUB")
    for elem in node.body: self.visit(elem)
    for elem in node.orelse: self.visit(elem)

  def collect_constraints(self):
    # print("Num nodes: ", len(self.call_graph.nodes))
    for n in self.call_graph.nodes:
      # print("Node ",n)
      node = self.function_map[n]
      self.curr_func = n
      super(RefImmutabilityAnalyzer, self).visit(node)

  def solve_constraints(self):
    while not (self.worklist == []):
      cons = self.worklist.pop()
      #print("\nPopped up constraint: ",cons)
      #cons.print_constraint()
      if cons.kind == 'EQU':
        successors = self._solve_EQU(cons.lhs,cons.rhs)
      else:
        successors = self._solve_SUB(cons.lhs,cons.rhs)
      for succ in successors:
        if not (succ in self.worklist):
          self.worklist.append(succ)

  def _is_field(self,side):
    assert ins(side,str)
    if ":" in side: return False
    else: return True

  def _get_type(self,side):
    if side == 'mutable':
      return 'mutable'
    elif ins(side,str) and not ":" in side:
      return self.attr_to_type[side]
    elif ins(side,tuple):
      # Viewpoint adaptation
      adapter_type = self._get_type(side[0])
      adaptee_type = self._get_type(side[1])
      if adaptee_type == 'mutable' or adaptee_type == 'readonly':
        return adaptee_type
      else: 
        assert adaptee_type == 'poly'
        return adapter_type
    else:
      return self.node_to_type[side]

  def _is_subtype(self,lhs,rhs):
    if lhs == 'mutable': return True
    elif rhs == 'readonly': return True
    elif lhs == 'poly' and rhs == 'readonly': return True
    elif lhs == 'poly' and rhs == 'poly': return True
    else: return False

  def _is_in_whitelist(self,node):
    if ins(node.func,ast.Attribute) and node.func.attr in self.unresolved_whitelist:
      return True
    return False

  def _is_in_blacklist(self,node):
    if ins(node.func,ast.Attribute) and node.func.attr in self.unresolved_blacklist:
      return True
    return False

  def _is_self_adapter(self, side):
    if ins(side,tuple) and ins(side[0],str) and side[0].endswith(":self"):
      return True
    else:
      return False
  
  def _is_self(self, side):
    if ins(side,str) and side.endswith(":self"): 
      return True
    else:
      return False

  def _propagate_self_locs(self, lhs, rhs):
    successors = []
    if ins(rhs,tuple) and self._is_self(rhs[1]):
      if rhs[1] in self.self_locs:
        if self._is_self(lhs): # self <: call |> self, call through self.
          change = False
          for loc in self.self_locs[rhs[1]]:
            if not self._in_map(self.self_locs,lhs,loc):
              self._add_to_map(self.self_locs,lhs,loc)
              change = True
          if change:
            successors = successors + self.node_to_constraints[lhs]
        else:
          if self.node_to_type[lhs] != 'mutable':
            self.node_to_type[lhs] = 'mutable'
            successors = successors + self.node_to_constraints[lhs] 
    return successors

  # returns successor constraints if change
  def _solve_SUB(self, lhs, rhs):
    successors = self._propagate_self_locs(lhs, rhs)
    if self._is_subtype(self._get_type(lhs),self._get_type(rhs)):
      return successors
    else:
      rhs_type = self._get_type(rhs)
      if (ins(lhs,str) or ins(lhs,ast.AST)) and self._get_type(lhs) != 'mutable':
        self.node_to_type[lhs] = rhs_type
        successors = self.node_to_constraints[lhs]
        # print("Change, adding constraints")
        # print(lhs, 'is now ',rhs_type)
      else: 
        assert ins(lhs,tuple)
        adapter = lhs[0]
        adaptee = lhs[1]
        if self._get_type(adaptee) != 'poly':
          # print("Change, adding constraints type of ", adaptee, "was ",self._get_type(adaptee))
          if self._is_field(adaptee):
            successors = successors + self.attr_to_constraints[adaptee]
          else:
            successors = successors + self.node_to_constraints[adaptee]
        if self._get_type(adapter) != rhs_type or self._get_type(adapter) != 'mutable':
          # print("Change, adding constraints 2")
          successors = successors + self.node_to_constraints[adapter]
        if self._is_field(adaptee):
          self.attr_to_type[adaptee] = 'poly'
        else:
          self.node_to_type[adaptee] = 'poly'
        self.node_to_type[adapter] = rhs_type
        # print(adaptee, 'is now POLY')
        # print(adapter, 'is now ',rhs_type)
      return successors       
  
  def _solve_EQU(self, lhs, rhs):
    successors = self._solve_SUB(lhs, rhs) + self._solve_SUB(rhs, lhs)
    return successors

class FragmentRefImmutabilityAnalyzer(RefImmutabilityAnalyzer):
  def __init__(self,call_graph,unresolved,libcalls,function_map,node_to_type,full_self_locs,starting_node):
    super(FragmentRefImmutabilityAnalyzer,self).__init__(call_graph,unresolved,libcalls,function_map)
    self.full_node_to_type = node_to_type
    self.full_self_locs = full_self_locs
    self.starting_node = starting_node
    self.mod_local = []

  def collect_constraints(self):
    super(FragmentRefImmutabilityAnalyzer, self).visit(self.starting_node)
  
  def _Actual_Formal_Assign(self, act, formal, call, callee_kind):
    # actual is an Expression, formal is a full string, call is a Call
    if callee_kind == 'UNRESOLVED' or callee_kind == 'LIBCALL':
      #print("here in framgment actual to formal assign")
      super(FragmentRefImmutabilityAnalyzer, self)._Actual_Formal_Assign(act,formal,call,callee_kind)
    else: 
      if self._is_self(formal) and formal in self.full_self_locs.keys(): #if formal is self, then we have to propagate self locs.
        self.self_locs[formal] = self.full_self_locs[formal]
      self.result_stack.append("MARKER")
      self.visit(act)
      lh_sides = []
      while self.result_stack[-1] != "MARKER":
        lh_sides.append(self.result_stack.pop())
      if formal in self.full_node_to_type.keys():
        formal_type = self.full_node_to_type[formal]
      else:
        formal_type = 'readonly'
      for lhs in lh_sides:
        if formal_type == 'mutable':
          self._add_Constraint(lhs,'mutable',"SUB")
        elif formal_type == 'poly':
          self._add_Constraint(lhs,call,'SUB')
        elif self._is_self(formal): # TODO: check this out
          self._add_Constraint(lhs,(call,formal),"SUB")
      self.result_stack.pop()

  def _Return_Node(self, node, return_nodes, callee_str, callee_kind):
    if callee_kind == 'UNRESOLVED' or callee_kind == 'LIBCALL':
      super(FragmentRefImmutabilityAnalyzer, self)._Return_Node(node,return_nodes,callee_str,callee_kind)
    else:
      ret_var_type = self.full_node_to_type[callee_str+":ret_var"]
      if ret_var_type == 'poly':
        return_nodes.append(node)

  def _Assign_Helper(self, lhs, rhs):
    if ins(lhs,ast.Name):
      name = self.curr_func+":"+lhs.id 
      if name not in self.mod_local: self.mod_local.append(name)
    super(FragmentRefImmutabilityAnalyzer, self)._Assign_Helper(lhs,rhs)

  #def visit_Assign(self, node):
  #  for target in node.targets: 
  #    self._Assign_Helper(target)
  #  self.generic_visit(node)
 
  #def visit_AugAssign(self, node):
  #  self._Assign_Helper(node.target)
  #  self.generic_visit(node)

  #def visit_AnnAssign(self, node):
  #  self._Assign_Helper(node.target)
  #  self.generic_visit(node)
        
  def get_mod_set(self):
    result = []
    result = result + self.mod_local
    for key in self.node_to_type:
      if self.node_to_type[key] == 'mutable' and ins(key,str):
        result.append((key,'*'))
    curr_self = self.curr_func+':self'
    if curr_self in self.self_locs.keys():
      for self_loc in self.self_locs[curr_self]:
        result.append((curr_self,self_loc))
    return result

class ReadAnalyzer(ast.NodeVisitor):

   # TODO: Probably have to add handling of implication...

   def __init__(self,starting_node,curr_func,libcalls):
     self.result = []
     self.starting_node = starting_node
     self.curr_func = curr_func
     self.libcalls = libcalls
     self.path = []

   def collect_read_set(self):
     self.visit(self.starting_node)     

   def _is_in_component(self):
     num_components = 0
     attr = None
     for tup in reversed(self.path):
       if tup[0] == 'Component':
         num_components += 1
         attr = tup[1]
     return (num_components,attr)

   def _is_in_call(self):
     for tup in reversed(self.path):
       if tup[0] == 'Call':
         return True
     return False         

   def visit_Name(self,node):
     is_in_call = self._is_in_call()
     (num_components, attr) = self._is_in_component()
     #print("Name: ",node.id,(num_components,attr))
     name = self.curr_func+":"+node.id
     if name not in self.result: self.result.append(name)
     if is_in_call or num_components >= 2:
       if (name,'*') not in self.result: self.result.append((name,"*"))
     elif not is_in_call and num_components == 1:
       if (name,attr) not in self.result: self.result.append((name,attr))
     

   def visit_Attribute(self,node):
     #print("Visiting Attribute node: ",ast.unparse(node))
     self.path.append(("Component",node.attr))
     self.visit(node.value)
     self.path.pop()

   def visit_Subscript(self,node):
     self.path.append(("Component","[]"))
     self.visit(node.value)
     self.path.pop()
  
   def visit_Call(self,node):
     #print("\nReadSet Visiting Call: "+ast.unparse(node));
     #print("Call ast dump: "+ast.dump(node))
     self.path.append(("Call",None))
     #if self.curr_func+":"+ast.unparse(node) in self.libcalls or ins(node.func,ast.Name):
     for arg in node.args:
       self.visit(arg)
     for keyword in node.keywords:
       if hasattr(keyword,'value') and (not ins(keyword, ast.Constant)):
       #if hasattr(keyword,'value') and keyword.value:
         self.visit(keyword.value)
     if not self.curr_func+":"+ast.unparse(node) in self.libcalls:
       if ins(node.func,ast.Name):
         pass
       elif ins(node.func,ast.Attribute):
         self.visit(node.func.value)
       else:
         self.visit(node.func)
     self.path.pop()

   
   def _is_name_or_self(self,target):
     if ins(target,ast.Name) or (ins(target,ast.Attribute) and ins(target.value,ast.Name) and target.value.id=='self'):
       return True
     else:
       return False

   def visit_Assign(self,node):
     self.visit(node.value)
     for target in node.targets: 
       if not self._is_name_or_self(target):
         self.visit(target)

   def visit_AugAssign(self,node):
     self.visit(node.value)
     if not self._is_name_or_self(node.target):
       self.visit(node.target)   

   def visit_AnnAssign(self,node):
     self.visit(node.value)
     if not self._is_name_or_self(node.target):
       self.visit(node.target)

# This one needs testing!
def intersect_mod_read(pre_mod_set,pre_read_set):
    # Remove the naming path. i.e. from ('/Users/.../AppData/Local/.../feature_extraction/text.py:CountVectorizer:fit_transform:self', 'vocabulary')
    # to ('self', 'vocabulary')
    mod_set = []
    read_set = []
    for mod in pre_mod_set:
      if ins(mod,str):
        mod_set.append(mod.split(":")[-1])
      else:
        assert ins(mod,tuple)
        mod_set.append((mod[0].split(":")[-1], mod[1]))

    for read in pre_read_set:
      if ins(read,str):
        read_set.append(read.split(":")[-1])
      else:
        assert ins(read,tuple)
        read_set.append((read[0].split(":")[-1], read[1]))

    #for loc in mod_set: print("---MOD--- ",loc) 
    #for loc in read_set: print("---READ--- ",loc) 

    # mod_set and read_set contain entries of the form:
    # 1. loc (e.g., X, self, var) meaning that there is a definition of loc in code fragment that gives rise to mod_set 
    # For read_set, it means the value of loc is read in formula Q that gives rise to 
    # 2. loc.* (e.g., X.*, self.*) meaning that some location transitively reachable from X may be modified (read)
    # 3. loc.f (e.g., self.n_features, self.estimator) meaning that exactly that location may be modified (read) 
    for read in read_set:
     if ins(read,str):
       if read in mod_set:
         return True
         # since there is a def in stmt and a use in Q, {Q} stmt {Q} is UNSOUND
       # otherwise, if read is not in mod_set, then all is good
     else:
       assert ins(read,tuple)
       if read[0] == 'self' and read[1] == '*': #(self.*)
         for mod in mod_set:
           if ins(mod,tuple) and mod[0] == 'self': #intersects if there is self, self.* or self.attr in mod 
             return True
       elif read[0] == 'self': #(self.attr)
         for mod in mod_set:
           if ins(mod,tuple) and mod[0] == 'self' and (mod[1] == '*' or mod[1] == read[1]): #intersects if self.* or self.attr (exactly) in mod set
             return True
       elif read[1] == '*': #(loc.*) where loc != self
         for mod in mod_set:
           if ins(mod,tuple) and mod[0] != 'self': #intersects if there is loc'.* or loc'.attr. Any loc.attr and loc'.attr' may be aliased
             return True
       else: #(loc.attr)
         for mod in mod_set:
           if ins(mod,tuple) and mod[0] != 'self' and mod[1] == read[1]:
             return True
    return False

def main(operator_main):

   # UNUSED CODES BELOW. FOR TESTING AND STUFF
   
   operators = ['PCA','ExtraTreesClassifier','DecisionTreeClassifier','KNeighborsClassifier','LogisticRegression','GradientBoostingClassifier']
   entry_points = {'PCA':['fit'],'ExtraTreesClassifier':['fit'],'DecisionTreeClassifier':['fit'],'KNeighborsClassifier':['fit'],'LogisticRegression':['fit'],'GradientBoostingClassifier':['fit']}   

   #pandas
   #operators = ['DataFrame','Series']
   #entry_points = {'DataFrame':['__init__']} #,'to_records','to_stata']}

   #lightgbm
   #operators = ['LGBMRegressor','LGBMClassifier','LGBMRanker']
   #entry_points = {'LGBMRegressor':['fit'],'LGBMClassifier':['fit'],'LGBMRanker':['fit']}

   #xgboost
   #operators = ['XGBClassifier','XGBRegressor','XGBRanker']                                                                           
   #entry_points = {'XGBClassifier':['fit'],'XGBRegressor':['fit'],'XGBRanker':['fit']} 

   #dummy
   #operators = ['Dummy']
   #entry_points = {'Dummy':['fit5']}

   function_map, bases_map, imports_map, globals_map = REF_crawler.get_function_map()

   operator = operators[4] 
   print("\n\n Analyzing operator: ", operator)
   call_graph_analyzer = call_graph.CallGraphAnalyzer(function_map,bases_map,imports_map,globals_map);
   for entry_point in entry_points[operator]: 
     call_graph_analyzer.solve_worklist(operator,entry_point)

   call_graph_analyzer.call_graph.printGraph()
   print("\n Num calls: ",call_graph_analyzer.num_calls," and num UNRESOLVED ",call_graph_analyzer.num_unresolved, " and num LIBCALLS ", call_graph_analyzer.num_libcalls) 
   print("A DAG: ",call_graph_analyzer.call_graph.isDAG())

   immutability_analyzer = RefImmutabilityAnalyzer(call_graph_analyzer.call_graph,call_graph_analyzer.unresolved,call_graph_analyzer.libcalls,function_map)
   immutability_analyzer.collect_constraints()
   immutability_analyzer.solve_constraints()
   #print("Printing constraints: ")
   #for c in immutability_analyzer.constraints:
   #  c.print_constraint()
   #print("Printing self.self_locs: ")
   #for key in immutability_analyzer.self_locs:
   #  print("\n Self locs for ", key, " of type ", immutability_analyzer.node_to_type[key])
   #  for loc in immutability_analyzer.self_locs[key]:
   #    print("-- ",loc)

   #print("Printing self.node_to_type: ")
   #for key in immutability_analyzer.node_to_type:
   #  if not ins(key,str): key1 = ast.unparse(key)
   #  else: key1 = key
   #  print("\n Type for ", key1, " is ", immutability_analyzer.node_to_type[key])


   print("\n\n\n =============================================== \n\n\n")

   
   # Uncomment to run Fragment Analyzer on all stmts in all reachable methods
   if 0:
     for func_name in call_graph_analyzer.call_graph.nodes:
       func = function_map[func_name]
       for stmt in func.body:   
         if True: # ins(stmt,ast.For):
           print("\n\nFragment analyzer for ",func.name," and ",ast.unparse(stmt))
           fragment_immutability_analyzer = FragmentRefImmutabilityAnalyzer(call_graph_analyzer.call_graph,call_graph_analyzer.unresolved,call_graph_analyzer.libcalls,function_map,immutability_analyzer.node_to_type,immutability_analyzer.self_locs,stmt)
           fragment_immutability_analyzer.curr_func = func_name
           fragment_immutability_analyzer.collect_constraints()
           fragment_immutability_analyzer.solve_constraints()
           #print("Printing node_to_type: ")
           #for key in fragment_immutability_analyzer.node_to_type:                                                                  
           #  if ins(key,str): 
           #    print("Type for ", key, " is ", fragment_immutability_analyzer.node_to_type[key])  
           mod_set = fragment_immutability_analyzer.get_mod_set()  
           print("Printing mod set: ")
           for loc in mod_set: print("--- ",loc)
   
   # Running Read Analyzer on all stmts in all reachable methods
   if 0:
     for func_name in call_graph_analyzer.call_graph.nodes:                                                                                 
        func = function_map[func_name]                                                                                           
        for stmt in func.body:
          if True: #ins(stmt,ast.For):
            print("\n\nRead analyzer for ",func.name," and ",ast.unparse(stmt))
            read_analyzer = ReadAnalyzer(stmt,func_name,call_graph_analyzer.libcalls)
            read_analyzer.collect_read_set()
            print("Printing read set: ")
            for loc in read_analyzer.result: print("--- ", loc)   
 
if __name__ == "__main__":
  print("> REF_ref_immutability.py: NOTHING IS HERE")


    
