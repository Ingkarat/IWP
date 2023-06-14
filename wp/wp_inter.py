#!/usr/local/bin/python3.9

"""
Note:
- Specific case for dealing with Dict() in svm/_base.py:_get_liblinear_solver_type()
   Currently, we detect this specific exception at the end and replace the dict with the disjunction of conjunctions.
- Comparing ast.Set: assume that all elemennts are Constant(value='abc')
"""

from timeit import default_timer as timer

import ast
import astor
import pickle

import sys

import crawler
import call_graph
import wp_intra
import pprint2
import equality
import helper
import config

#import AA_call_graph_alone

import REF_call_graph
import REF_crawler
import REF_graph
import REF_ref_immutability
import REF_symbol_table

from simplify import TRUE
from equality import Impl

# get function map
# get call_graph

# initialize main class and main func

# get reversed DAG
# for each func in reversed DAG
#   1. propagate Raises wp_script: call visitFunctionDef with Raise and False
#   2. for each callee in call graph, propagate exceptions in callee into caller: 
#        for each exception in the wps[callee] substitute, then save into wps[caller]

def rpl(x):
   return x.replace(config.PATH_SHORTENING,"")


contain = False
def containFilteredASTConstant(wp):
  cc = 0
  if isinstance(wp,ast.Constant):
    if isinstance(wp.value,str):
        #print(">",ast.dump(wp))
        if "FILTERED WP. IT HAS" in wp.value:
            global contain
            contain = True
  else:
    for attr in wp.__dict__.keys():
      if isinstance(wp.__dict__[attr],ast.AST):
        containFilteredASTConstant(wp.__dict__[attr])
      elif isinstance(wp.__dict__[attr],list):
        for elt in wp.__dict__[attr]:
          containFilteredASTConstant(elt)

global_start_time = 0

global_f_prune = None
global_check_array = False
global_disable_staticCE = False
global_pruning_exp = False
global_chk_ary = False
global_chk_ary_name = ""

# f_call_graph_ver: 1 = sklearn, 2 = lightgbm
def main(package_dir, package_name, class_name, function_name, XXXXXX):

   pickleWpToFile = False  # WP result to pickle file
   writeToFile = False  # WP result (and some info) to text file

   # ref immutability
   REF_function_map, REF_bases_map, REF_imports_map, REF_globals_map = REF_crawler.get_function_map(package_dir, package_name)
   REF_call_graph_analyzer = REF_call_graph.CallGraphAnalyzer(REF_function_map,REF_bases_map,REF_imports_map,REF_globals_map)
   REF_call_graph_analyzer.solve_worklist(class_name,function_name)
   if config.PRINT_DEBUG:
      REF_call_graph_analyzer.call_graph.printGraph()
      print("\n Num calls: ",REF_call_graph_analyzer.num_calls," and num UNRESOLVED ",REF_call_graph_analyzer.num_unresolved, " and num LIBCALLS ", REF_call_graph_analyzer.num_libcalls) 
      print("A DAG: ",REF_call_graph_analyzer.call_graph.isDAG())

   immutability_analyzer = REF_ref_immutability.RefImmutabilityAnalyzer(REF_call_graph_analyzer.call_graph,REF_call_graph_analyzer.unresolved,REF_call_graph_analyzer.libcalls,REF_function_map)
   immutability_analyzer.collect_constraints()
   immutability_analyzer.solve_constraints()

   #print("\n\n\n ================== END OF GLOBAL REF IMMU ================== \n\n\n")

   graph_analyzer = call_graph.main(package_dir, class_name, function_name)
   function_map = graph_analyzer.function_map
   base_map = graph_analyzer.bases_map

   reversed_DAG = call_graph.reverseGraph(graph_analyzer.call_graph)
   reversed_topo = graph_analyzer.call_graph.reversedTopo() # have to call isDAG2() beforehand. In this case it's called in call_graph.main()

   if len(reversed_topo) == 0:  # Add main_func if call graph is empty
      reversed_topo.append(graph_analyzer.main_func)

   if config.PRINT_SOME_INFO:
   #print("\n\n============== wp_inter.py ==============")
      print("Reversed Topo (order of analyzing):")
      for rr in reversed_topo:
         print("> ", rpl(rr))

   # Function that return some of unmodified parameters. Eg. _check_solver
   if package_name == "sklearn":
      helper_function = helper.checkSpecialFunction(function_map)
   else:
      helper_function = {}

   # TODO: revisit comments in this file. They are from previous iteration
   
   # initializes the Function visitor with the function name and an empty map for now. So just propagates raises intraprocedurally
   # The idea is that we'll analyze Calls and propagate the raises in callee function up the caller
   # The map argument will be a map from a Call ast node to a WP formula _after_ substitution of actuals for formals
   # FunctionDef will take that formula and propagate intraprocedurally   
   # ...: Have to change the wp map in Analyzer... Currently it maps each node to a SINGLE WP formula. But a function can have more than one formulas
   # have to figure out how to propagate those

   # Probably best to run first with func,{} to get all immediate exceptions raised in func. 
   # Then look at call graph and get exceptions for callees. For each one initialize with func, {Call ast, wp_formula} one by one.
   # The call graph gives you info about the Call ast node, so we can record immediately

   if config.PRINT_DEBUG:
      for rt in reversed_topo:
         print("> At node:",rpl(rt))
         call_edge = reversed_DAG.getEdgesFromSource(rt)
         for ce in call_edge:
            print("S = ",rpl(ce.src))
            print("T = ",rpl(ce.tgt))
            print("Label = ",ast.dump(ce.label))


   # Map func name to wp_intra.Analyzer
   analyzer_map = {}
   # List of analyers
   analyzer_list = []
   skip_selfloop = False


   for rt in reversed_topo:
      
      if config.PRINT_SOME_INFO: 
         print("\n\n************ Working At node:",rpl(rt))

      all_kwarg_default = {}
      callee_ALL_wps = {}
      callee_REF_soundness_flag = {}
      call_edges = reversed_DAG.getEdgesToTarget(rt)
      if config.PRINT_SOME_INFO: 
         print("Len is",len(call_edges))
      for ce in call_edges:
         callee = ce.src
         if config.PRINT_SOME_INFO: 
            print(">")
            print(ce.src)
            print(ce.tgt)
            print(ast.dump(ce.label))
         # get analyzer for callee (ce.src). There can be only 1 analyzer for each function
         callee_analyzer = None
         for al in analyzer_list:
            if al.func_name == ce.src:
               callee_analyzer = al


         callee_wps = callee_analyzer.wps
         #print("__")
         #print(callee_wps)
         
         for key in callee_wps.keys():
            key_added = key
            key_append = [ce.label, rt, callee] # call node, caller f name, callee f name
            key_added += tuple(key_append)
            callee_ALL_wps[key_added] = callee_wps[key]
            callee_REF_soundness_flag[key_added] = callee_analyzer.REF_soundness_flag[key]

         for key in callee_analyzer.kwargDefault.keys():
            key_added = key
            key_append = [ce.label, rt, callee] # call node, caller f name, callee f name
            key_added += tuple(key_append)
            all_kwarg_default[key_added] = callee_analyzer.kwargDefault[key]


      intra_analyzer = wp_intra.Analyzer(rt, callee_ALL_wps, function_map, all_kwarg_default, helper_function
         , REF_call_graph_analyzer, REF_function_map, immutability_analyzer, callee_REF_soundness_flag)
      intra_analyzer.visit_FunctionDef(function_map[rt])

      # Heuristic (hack!?) for the specific exeption
      # Simplify it to be >>>   sp.issparse(array)  =>  NOT(accept_sparse Is False)   <<<
      if rpl(rt) == "/utils/validation.py:None:check_array":
         asttt = ast.BoolOp(op=Impl(), values=[ast.Call(func=ast.Attribute(value=ast.Name(id='sp', ctx=ast.Load()), attr='issparse', ctx=ast.Load()), args=[ast.Name(id='array', ctx=ast.Load())], keywords=[]), ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=ast.Name(id='accept_sparse', ctx=ast.Load()), ops=[ast.Is()], comparators=[ast.Constant(value=False)]))])
         for wp in intra_analyzer.wps.keys():
            if isinstance(wp[0], ast.Raise):
               if isinstance(wp[0].exc, ast.Call):
                  if isinstance(wp[0].exc.args[0], ast.Constant):
                     if "A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array." in wp[0].exc.args[0].value:
                        intra_analyzer.wps[wp] = asttt

      analyzer_list.append(intra_analyzer)




   print("\n\n============= Displaying results =============")

   ONLY_SPARSE_EXCEPTION = False
   
   #TODO: THIS
   f_simplify = True
   opName = "???"

   if f_simplify:
      if global_f_prune:
         outputName1 = "output_time/wp/CE/["+opName+"]"+"NoTrue"
         outputName2 = "output_time/wp/CE/["+opName+"]"+"True"
      else:
         # f_prune = False
         outputName1 = "output_time/wp/CEnoP/["+opName+"]"+"NoTrue"
         outputName2 = "output_time/wp/CEnoP/["+opName+"]"+"True"
   else:
      if not global_f_prune:
         outputName1 = "output_time/wp/noCE/["+opName+"]"+"NoTrue"
         outputName2 = "output_time/wp/noCE/["+opName+"]"+"True"
      else:
         # f_prune = True
         outputName1 = "output_time/wp/PnoCE/["+opName+"]"+"NoTrue"
         outputName2 = "output_time/wp/PnoCE/["+opName+"]"+"True"

   if global_check_array:
      if f_simplify:
         if global_f_prune:
            outputName1 = "output_time/check_array/wp/CE/["+opName+"]"+"NoTrue"
            outputName2 = "output_time/check_array/wp/CE/["+opName+"]"+"True"
         else:
            # f_prune = False
            outputName1 = "output_time/check_array/wp/CEnoP/["+opName+"]"+"NoTrue"
            outputName2 = "output_time/check_array/wp/CEnoP/["+opName+"]"+"True"
      else:
         if not global_f_prune:
            outputName1 = "output_time/check_array/wp/noCE/["+opName+"]"+"NoTrue"
            outputName2 = "output_time/check_array/wp/noCE/["+opName+"]"+"True"
         else:
            # f_prune = True
            outputName1 = "output_time/check_array/wp/PnoCE/["+opName+"]"+"NoTrue"
            outputName2 = "output_time/check_array/wp/PnoCE/["+opName+"]"+"True"

   if global_disable_staticCE:
      outputName1 = "output_time/staticCE/wp/["+opName+"]"+"NoTrue"
      outputName2 = "output_time/staticCE/wp/["+opName+"]"+"True"

   if global_pruning_exp:
      outputName1 = "output_time/pruning_exp/wp/["+opName+"]"+"NoTrue"
      outputName2 = "output_time/pruning_exp/wp/["+opName+"]"+"True"  

   if global_chk_ary:
      outputName1 = "output_time/chk_ary/wp/["+opName+"]"+global_chk_ary_name+"NoTrue"
      outputName2 = "output_time/chk_ary/wp/["+opName+"]"+global_chk_ary_name+"True"  



   numTrue = 0
   numNotTrue = 0
   numSoundNotTrue = 0
   numUnsoundNotTrue = 0
   main_wps_cut = {}
   main_has_CA = {} # exception's path contain check_array()
   main_no_CA = {} # exception's path DOES NOT contain check_array()

   numHasCATrue = 0
   numHasCANotTrue = 0
   numHasCASoundNotTrue = 0
   numHasCAUnSoundNotTrue = 0
   numNoCATrue = 0
   numNoCANotTrue = 0
   numNoCASoundNotTrue = 0
   numNoCAUnSoundNotTrue = 0

   #TODO: THIS
   outputName1 = "/Users/ingkarat/Documents/GitHub/IWP/output/a"
   outputName2 = "/Users/ingkarat/Documents/GitHub/IWP/output/b"

   with open(outputName1, "w") as f1, open(outputName2,"w") as f2:
      main_analyzer = analyzer_list[-1]
      main_wps = main_analyzer.wps
      count = 0
      for wp in main_wps.keys():
         print("\n",count+1)
         print("[IMPL COUNT]:",helper.impl_counter(main_wps[wp]))
         print("[SN_FLAG]: ", main_analyzer.REF_soundness_flag[wp])
         if helper.impl_counter(main_wps[wp]) <= 200:
            print("[WP]:\n\t", pprint2.pprint_top(main_wps[wp]))
         else:
            print("[WP]:\n\t(WP too long. Did not print)")
         print("[Raise node] at:", rpl(wp[1]), "\n\t", ast.dump(wp[0]),"\n")

         if helper.impl_counter(main_wps[wp]) <= 200:
            main_wps_cut[wp] = main_wps[wp]
         else:
            #count += 1
            if global_check_array:
               main_wps_cut[wp] = main_wps[wp]
            if global_chk_ary:
               main_wps_cut[wp] = main_wps[wp]
            else:
               count += 1
               continue

         global contain
         contain = False
         containFilteredASTConstant(main_wps_cut[wp])
         if contain:
            #if not global_check_array:
            #   continue
            if not global_chk_ary:
               continue
            #print(ast.dump(main_wps[wp]))
         #print("=====")
         #print(ast.dump(main_wps[wp]))
      

         if 0:
            if (not equality.compare(main_wps[wp],TRUE)):
               f = f1
               numNotTrue += 1
            else:
               f = f2
               numTrue += 1
         thisIsTrue = 0
         thisIsNotTrue = 0
         thisIsSoundNotTrue = 0
         thisIsUnsoundNotTrue = 0
         if (not equality.compare(main_wps_cut[wp],TRUE)):
            f = f1
            numNotTrue += 1
            if main_analyzer.REF_soundness_flag[wp]:
               numSoundNotTrue += 1
               thisIsSoundNotTrue = 1
            else:
               numUnsoundNotTrue += 1
               thisIsUnsoundNotTrue = 1
            thisIsNotTrue = 1 
         else:
            f = f2
            numTrue += 1
            thisIsTrue = 0


         containCheck_Array = False

         if writeToFile:
            print("\n",count+1,file = f)
            print("[IMPL COUNT]:",helper.impl_counter(main_wps_cut[wp]),file = f)
            print("[SN_FLAG]: ", main_analyzer.REF_soundness_flag[wp],file = f)
            #print("[WP]:\n\t", pprint.pprint_top(main_wps[wp]),file = f)
            print("[WP]:\n\t", pprint2.pprint_top(main_wps_cut[wp]),file = f)
            print("[Raise node] at:", rpl(wp[1]), "\n\t", ast.dump(wp[0]),"\n", file = f)

            if "/utils/validation.py:None:check_array" in wp[1]:
               containCheck_Array = True

         num = len(wp) - 3
         while num > 1:
            print("FROM: ", rpl(wp[num+1]),"\tTO:", rpl(wp[num+2]))
            print("[Call node]:\n\t", ast.dump(wp[num]))
            print("[Call code] (using ast.unparse):\n\t", ast.unparse(wp[num]),"\t")
            if writeToFile:
               print("FROM: ", rpl(wp[num+1]),"\tTO:", rpl(wp[num+2]), file = f)
               print("[Call node]:\n\t", ast.dump(wp[num]), file = f)
               print("[Call code] (using ast.unparse):\n\t", ast.unparse(wp[num]),"\t", file = f)

            if "/utils/validation.py:None:check_array" in wp[num+1]:
               containCheck_Array = True
            if "/utils/validation.py:None:check_array" in wp[num+2]:
               containCheck_Array = True
            num -= 3

         if containCheck_Array:
            main_has_CA[wp] = main_wps_cut[wp]
            numHasCATrue += thisIsTrue
            numHasCANotTrue += thisIsNotTrue
            numHasCASoundNotTrue += thisIsSoundNotTrue
            numHasCAUnSoundNotTrue += thisIsUnsoundNotTrue
         else:
            main_no_CA[wp] = main_wps_cut[wp]
            numNoCATrue += thisIsTrue
            numNoCANotTrue += thisIsNotTrue
            numNoCASoundNotTrue += thisIsSoundNotTrue
            numNoCAUnSoundNotTrue += thisIsUnsoundNotTrue


         count += 1

      assert False, "SO FAR SO GOOD (WORKING ON RESULT PATHS)"

      main_kwargDefaults = main_analyzer.kwargDefault
      count = 0
      for k in main_kwargDefaults.keys():
         if len(main_kwargDefaults[k]) > 0:
            print("\n",count+1)
            print(main_kwargDefaults[k])
         if 0:
            num = len(wp) - 3
            while num > 1:
               print("FROM: ", rpl(wp[num+1]),"\tTO:", rpl(wp[num+2]))
               print("[Call node]:\n\t", ast.dump(wp[num]))
               print("[Call code] (using ast.unparse):\n\t", ast.unparse(wp[num]),"\t")
               num -= 3
         count += 1


   if f_simplify:
      if global_f_prune:
         outputName3 = "output_time/pkl/CE/" + opName + ".pkl"
      else:
         # f_prune == False
         outputName3 = "output_time/pkl/CEnoP/" + opName + ".pkl"
   else:
      if not global_f_prune:
         outputName3 = "output_time/pkl/noCE/" + opName + ".pkl"
      else:
         # f_prune == True
         outputName3 = "output_time/pkl/PnoCE/" + opName + ".pkl"

   if global_check_array:
      if f_simplify:
         if global_f_prune:
            outputName3 = "output_time/check_array/pkl/CE/" + opName + ".pkl"
         else:
            # f_prune == False
            outputName3 = "output_time/check_array/pkl/CEnoP/" + opName + ".pkl"
      else:
         if not global_f_prune:
            outputName3 = "output_time/check_array/pkl/noCE/" + opName + ".pkl"
         else:
            # f_prune == True
            outputName3 = "output_time/check_array/pkl/PnoCE/" + opName + ".pkl"

   if global_disable_staticCE:
      outputName3 = "output_time/staticCE/pkl/" + opName + ".pkl"

   if global_pruning_exp:
      outputName3 = "output_time/pruning_exp/pkl/" + opName + ".pkl"

   if global_chk_ary:
      outputName3 = "output_time/chk_ary/pkl/" + opName + ".pkl"


   if global_chk_ary:
      pickleWpToFile = False

   if pickleWpToFile:
      with open(outputName3, "wb") as f3:
         main_analyzer = analyzer_list[-1]
         main_wps = main_analyzer.wps
         pickle.dump(main_wps, f3, pickle.HIGHEST_PROTOCOL)


   if f_simplify:
      if global_f_prune:
         outputNameStat = "output_time/stats/CE_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/stats/CE_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/stats/CE_sklearn_3_NoCA_stats"
      else:
         # f_prune == False
         outputNameStat = "output_time/stats/CEnoP_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/stats/CEnoP_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/stats/CEnoP_sklearn_3_NoCA_stats"
   else:
      if not global_f_prune:
         outputNameStat = "output_time/stats/noCE_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/stats/noCE_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/stats/noCE_sklearn_3_NoCA_stats"
      else:
         #f_prune == True
         outputNameStat = "output_time/stats/PnoCE_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/stats/PnoCE_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/stats/PnoCE_sklearn_3_NoCA_stats"  

   if global_check_array:
      if f_simplify:
         if global_f_prune:
            outputNameStat = "output_time/check_array/stats/CE_sklearn_1_All_stats"
            outputNameStatHasCA = "output_time/check_array/stats/CE_sklearn_2_HasCA_stats"
            outputNameStatNoCA = "output_time/check_array/stats/CE_sklearn_3_NoCA_stats"
         else:
            # f_prune == False
            outputNameStat = "output_time/check_array/stats/CEnoP_sklearn_1_All_stats"
            outputNameStatHasCA = "output_time/check_array/stats/CEnoP_sklearn_2_HasCA_stats"
            outputNameStatNoCA = "output_time/check_array/stats/CEnoP_sklearn_3_NoCA_stats"
      else:
         if not global_f_prune:
            outputNameStat = "output_time/check_array/stats/noCE_sklearn_1_All_stats"
            outputNameStatHasCA = "output_time/check_array/stats/noCE_sklearn_2_HasCA_stats"
            outputNameStatNoCA = "output_time/check_array/stats/noCE_sklearn_3_NoCA_stats"
         else:
            #f_prune == True
            outputNameStat = "output_time/check_array/stats/PnoCE_sklearn_1_All_stats"
            outputNameStatHasCA = "output_time/check_array/stats/PnoCE_sklearn_2_HasCA_stats"
            outputNameStatNoCA = "output_time/check_array/stats/PnoCE_sklearn_3_NoCA_stats"        

   if global_disable_staticCE:
         outputNameStat = "output_time/staticCE/stats/staticCE_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/staticCE/stats/staticCE_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/staticCE/stats/staticCE_sklearn_3_NoCA_stats"

   if global_pruning_exp:
         outputNameStat = "output_time/pruning_exp/stats/staticCE_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/pruning_exp/stats/staticCE_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/pruning_exp/stats/staticCE_sklearn_3_NoCA_stats" 

   if global_chk_ary:
         outputNameStat = "output_time/chk_ary/stats/staticCE_sklearn_1_All_stats"
         outputNameStatHasCA = "output_time/chk_ary/stats/staticCE_sklearn_2_HasCA_stats"
         outputNameStatNoCA = "output_time/chk_ary/stats/staticCE_sklearn_3_NoCA_stats"     


   with open(outputNameStat, "a+") as ffs:
      print("["+f_opName+","+str(numTrue+numNotTrue)+","+str(numTrue)+","+str(numNotTrue)+","+str(numSoundNotTrue)+","+str(numUnsoundNotTrue)+"]", file = ffs)

   with open(outputNameStatHasCA, "a+") as ffs:
      print("["+f_opName+","+str(numHasCATrue+numHasCANotTrue)+","+str(numHasCATrue)+","+str(numHasCANotTrue)+","+str(numHasCASoundNotTrue)+","+str(numHasCAUnSoundNotTrue)+"]", file = ffs)

   with open(outputNameStatNoCA, "a+") as ffs:
      print("["+f_opName+","+str(numNoCATrue+numNoCANotTrue)+","+str(numNoCATrue)+","+str(numNoCANotTrue)+","+str(numNoCASoundNotTrue)+","+str(numNoCAUnSoundNotTrue)+"]", file = ffs)


   if ONLY_SPARSE_EXCEPTION:
      with open(outputNameSUM, "a+") as fs:
         numTotal = numTrue + numNotTrue
         #"[NotTrue: " + str(numNotTrue) + "] "
         print("Total exceptions: " + str(numTotal).zfill(2) + "\t" + "True: " + str(numTrue).zfill(2) + "\t" + "NotTrue: " + str(numNotTrue)
               + "\t- " + f_opName + "(" + fname + ")", file = fs)




# get reversed DAG
# for each func in reversed DAG
#   1. propagate Raises wp_script: call visitFunctionDef with Raise and False
#   2. for each callee in call graph, propagate exceptions in callee into caller: 11
#        for each exception in the wps[callee] substitute, then save into wps[caller]

if __name__ == "__main__":

   #main(1,2,3,4,5)

   # we apply SIMPLIFY to all runs because it should be
   # 1 plain = no CE + no prune    in "noCE" folder
   # 2 CE + no prune
   # 3 CE + prune

   # f_prune = check def replace_in_formula() in wp_intra
   # f_simplify = Just go directly to def simplify() and dont call interpret_node_natively
   
   #f_name = "check_array"
   f_name = "fit"
   f_call_graph_ver = 1

   f_simplify = True # (mean CE) Currenly this flag only change the file name. There are too many places to apply. Just go directly to def simplify() and dont call interpret_node_natively
   f_prune = True
   disable_staticCE = False # staticCE = Simplify. This = no Simp, no CE, no Prune
   pruning_exp = False 
   global_check_array = False
   global_chk_ary = False
   global_chk_ary_name = ""
   # TODO
   # 1. fix PCA fit() = rewind to original
   # 2. fix check_array() = remove class

   global_f_prune = f_prune
   global_disable_staticCE = disable_staticCE
   global_pruning_exp = pruning_exp


   if f_simplify:
      # case with and without pruning
      if f_prune:
         time_path = "output_time/CE_sklearn_time.py"
         time_after_ref_path = "output_time/CE_sklearn_time_afterRef.py"
      else:
         time_path = "output_time/CEnoP_sklearn_time.py"
         time_after_ref_path = "output_time/CEnoP_sklearn_time_afterRef.py"
   else:
      if f_prune:
         time_path = "output_time/PnoCE_sklearn_time.py" 
         time_after_ref_path = "output_time/PnoCE_sklearn_time_afterRef.py" 
      else:
         time_path = "output_time/noCE_sklearn_time.py" 
         time_after_ref_path = "output_time/noCE_sklearn_time_afterRef.py" 

   if global_check_array:
      if f_simplify:
         if f_prune:
            time_path = "output_time/check_array/CE_sklearn_time.py"
            time_after_ref_path = "output_time/check_array/CE_sklearn_time_afterRef.py"
         else:
            time_path = "output_time/check_array/CEnoP_sklearn_time.py"
            time_after_ref_path = "output_time/check_array/CEnoP_sklearn_time_afterRef.py"
      else:
         if f_prune:
            time_path = "output_time/check_array/PnoCE_sklearn_time.py" 
            time_after_ref_path = "output_time/check_array/PnoCE_sklearn_time_afterRef.py" 
         else:
            time_path = "output_time/check_array/noCE_sklearn_time.py" 
            time_after_ref_path = "output_time/check_array/noCE_sklearn_time_afterRef.py"

   if global_disable_staticCE:       
      time_path = "output_time/staticCE/sklearn_time.py"
      time_after_ref_path = "output_time/staticCE/sklearn_time_afterRef.py"

   if global_pruning_exp:
      time_path = "output_time/pruning_exp/sklearn_time.py"
      time_after_ref_path = "output_time/pruning_exp/sklearn_time_afterRef.py"  

   if global_chk_ary:
      time_path = "output_time/chk_ary/sklearn_time.py"
      time_after_ref_path = "output_time/chk_ary/sklearn_time_afterRef.py"      


   ops = ['PCA']
   main("RidgeClassifier", "fit", f_skipUtilsValidation, f_simplify, f_call_graph_ver)
   
   global_chk_ary_name = "[NEW][...]"

   clsss = "PCA"


   if 0:
      for op in ops:
         try:
            start = timer()
            global_start_time = start
            main(op, f_name, f_skipUtilsValidation, f_simplify, f_call_graph_ver)
            end = timer()
            print(start, after_ref_timer, end)
            print("Time =",end - start) # Time in seconds, e.g. 5.38091952400282
            with open(time_path, "a+") as ft:
               print(op + " =",end - start, file = ft)
            with open(time_after_ref_path, "a+") as ftr:
               print(op + " =",end - after_ref_timer, file = ftr)
         except:
            with open("output_time/error.py", "a+") as ft:
               print(op, file = ft)
            pass
         print("\n\n> ", op)

   print("\n\n\n=== END ===")
