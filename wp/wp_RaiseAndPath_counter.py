import ast
from typing import Dict

import crawler
import call_graph

def rpl(x):
	return x.replace("/site-packages/sklearn","")

class RaiseAnalyzer(ast.NodeVisitor):
	def __init__(self):
		self.raise_counter = 0

	def visit_FunctionDef(self, node):
		#print("AT visit_FunctionDef:", node.name)
		self.generic_visit(node)

	def visit_Raise(self, node):
		self.raise_counter += 1

# Count all raise exceptions
def f_count_raise(graph_analyzer, reversed_DAG, reversed_topo) -> int:

	count = 0
	function_map = graph_analyzer.function_map
	raise_counter_map = {}

	for rt in reversed_topo:
		call_edges = reversed_DAG.getEdgesToTarget(rt)
		raise_analyzer = RaiseAnalyzer()
		raise_analyzer.visit_FunctionDef(function_map[rt])
		print("> ", rpl(rt), "=", raise_analyzer.raise_counter)
		count += raise_analyzer.raise_counter
		raise_counter_map[rt] = raise_analyzer.raise_counter

	return [count, raise_counter_map]


def f_count_path(graph_analyzer, reversed_DAG, reversed_topo, raise_counter_map) -> Dict[str, int]:
	
	combine_map = {}
	sanity_check = set()

	for rt in reversed_topo:
		call_edges = reversed_DAG.getEdgesToTarget(rt)
		print("\n\n************ Working At node:",rpl(rt))
		print("(base) raise_counter_map", raise_counter_map[rt])
		print("Len is",len(call_edges))

		base_raise = raise_counter_map[rt]
		summ = base_raise

		for ce in call_edges:
			print(">")
			print(rpl(ce.src))
			print(rpl(ce.tgt))
			print(ast.dump(ce.label))

			if ce.src == ce.tgt: # self call. Skip?
				continue

			assert not(ce.tgt in combine_map.keys()) # PANIC if we already visit this tgt (???)

			summ += combine_map[ce.src]

		combine_map[rt] = summ
		sanity_check.add(rt)

	print("{} {}".format(rpl(rt), combine_map[rt]))
	return combine_map


def main(op_name, f_name):

	graph_analyzer = call_graph.main(op_name, f_name)
	reversed_DAG = call_graph.reverseGraph(graph_analyzer.call_graph)
	reversed_topo = graph_analyzer.call_graph.reversedTopo() # have to call isDAG2() beforehand. In this case it's called in call_graph.main()
	
	if len(reversed_topo) == 0:
		reversed_topo.append(graph_analyzer.main_func)

	if 1:
		print("Reversed Topo:")
		for rr in reversed_topo:
			print("> ", rpl(rr))
		print("======= END Reversed Topo\n\n")

	ret = f_count_raise(graph_analyzer, reversed_DAG, reversed_topo)
	raise_count = ret[0]
	raise_counter_map = ret[1]

	combine_map = f_count_path(graph_analyzer, reversed_DAG, reversed_topo, raise_counter_map)

	print("\n\nTotal # of raise exceptions in the call graph starting from [{}'s {}()]".format(op_name, f_name), "\n>> ", raise_count)
	print("\n\n Total # of path to raise exceptions starting from [{}'s {}()]".format(op_name, f_name), "\n>> ", combine_map[reversed_topo[-1]])

if __name__ == "__main__":

	ops = ["LabelEncoder", "BernoulliRBM", "ComplementNB", "RFE", "MiniBatchKMeans", "TruncatedSVD", "SGDRegressor", "KernelPCA", "RandomForestRegressor", "RidgeClassifier"]

	op_name = "RidgeClassifier"
	f_name = "fit"
	
	main(op_name, f_name)

	print("=== END ===")