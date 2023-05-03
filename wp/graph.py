import ast

class Edge:
  def __init__(self, source, target, data):
    # src and tgt are strings, label is AST node ref
    self.src = source;
    self.tgt = target;
    self.label = data;

class Graph:
    def __init__(self):
        # rep invariants: 
        #    nodes has no duplicates
        #    edges.keys() is_subset nodes

        # nodes is a list of String function names
        self.nodes = []
        # adjacency list reprsentation from names to [Edge]
        # for each edge in edges[name] edge.src = name  
        self.edges = {}
        # topologically sorted order
        self.topo = []

    def hasNode(self,node):
        return node in self.nodes

    def addNode(self,node):
        if not self.hasNode(node):
         self.nodes.append(node)

    def hasEdge(self,source,target,data):
        if not (source in self.edges.keys()):
          return False
        edges_from_source = self.edges[source]
        for edge in edges_from_source:
          if edge.src == source and edge.tgt == target and edge.label == data:
            return True
        return False

    def addEdge(self,edge):
        if edge.src == edge.tgt:
            return

        #if "pandas/core/generic.py:NDFrame:all" in edge.src and "pandas/core/generic.py:NDFrame:_logical_func" in edge.tgt:
        #    return
        if "pandas/core/generic.py:NDFrame:_logical_func" in edge.src and "pandas/core/generic.py:NDFrame:all" in edge.tgt:
            return
        #if "pandas/core/generic.py:NDFrame:_logical_func" in edge.src and "pandas/core/frame.py:DataFrame:_reduce" in edge.tgt:
        #    return
        #if "pandas/core/dtypes/common.py:None:_is_dtype_type" in edge.src and "pandas/core/computation/ops.py:Op:is_scalar" in edge.tgt:
        #    return
        #if "pandas/core/tools/datetimes.py:None:_assemble_from_unit_mappings" in edge.src and "pandas/core/tools/datetimes.py:None:to_datetime" in edge.tgt:
        #    return
        if "pandas/core/dtypes/common.py:None:is_datetime64_dtype" in edge.src and "pandas/core/dtypes/common.py:None:_is_dtype_type" in edge.tgt:
            return
        if "pandas/core/frame.py:DataFrame:_reduce" in edge.src and "pandas/core/dtypes/common.py:None:is_object_dtype" in edge.tgt:
            return
        #if "pandas/core/generic.py:NDFrame:all" in edge.tgt:
        #    return

        self.addNode(edge.src)
        self.addNode(edge.tgt)
        if not self.hasEdge(edge.src,edge.tgt,edge.label):
          if not (edge.src in self.edges.keys()):
            self.edges[edge.src] = [edge]
          else:
            self.edges[edge.src].append(edge) 

    def getEdgesFromSource(self,source):
        if not (source in self.edges.keys()):
          return []
        else: 
          return self.edges[source]

    def getEdgesToTarget(self,target):
        result = []
        for source in self.edges.keys():
          for edge in self.edges[source]:
             if (edge.tgt == target):
               result.append(edge)
        return result

    def printGraph(self):
        replcs = "/pandas"
        print("\n Printing the graph (garph.py): ")    
        for key in self.edges.keys():
            for edge in self.edges[key]:
                print("Call from ",edge.src.replace(replcs,"")," to ",edge.tgt.replace(replcs,""))  

    def printGraphLabel(self):
        replc = "/sklearn"
        print("\n Printing the graph: ")    
        for key in self.edges.keys():
            for edge in self.edges[key]:
                if edge.src == edge.tgt:
                    print("\n\nSELF-LOOP ", edge.src)
                #print("Call from ",edge.src.replace(replc,"")," to ",edge.tgt.replace(replc,""),":")
                #print("__Label:",ast.dump(edge.label))
                #print(edge.label) 

    def reversedTopo(self):
        return self.topo

    #TODO: TEST!!!
    def isDAG(self):
        for key in self.edges.keys():
           worklist = []
           closure = []
           for edge in self.edges[key]:
              if not (edge.tgt in closure):
                closure.append(edge.tgt)
                worklist.append(edge.tgt)
           while not (worklist == []):
              node = worklist.pop()
              if node in self.edges.keys():
                for edge in self.edges[key]:
                  if not (edge.tgt in closure):
                    closure.append(edge.tgt)
                    worklist.append(edge.tgt)
           if key in closure:
             print("Call graph is NOT a DAG: ",key)
             return False 
        return True 

    def isDAG2(self): #NOT consider self-loop
        L = []
        N = []
        perm_mark = []
        temp_mark = []
        dag = True
        self.topo = None

        # remove 2 lines below so that we still get a list even though the graph is not DAG
        stillGetAListEvenIfNotDag = True

        def visit(n):
            nonlocal dag
            nonlocal L
            nonlocal perm_mark
            nonlocal temp_mark

            if not stillGetAListEvenIfNotDag:
                if not dag:
                    return
            if n in perm_mark:
                return
            if n in temp_mark:
                print("ISDAG2:", n)
                dag = False
                return

            temp_mark.append(n)

            for key in self.edges.keys():
                for edge in self.edges[key]:
                    if edge.src != edge.tgt:
                        if edge.src == n:
                            visit(edge.tgt)

            temp_mark.remove(n)
            perm_mark.append(n)
            L.append(n)

        for key in self.edges.keys():
            for edge in self.edges[key]:
                if edge.src != edge.tgt:
                    if edge.src not in N:
                        N.append(edge.src)
                    if edge.tgt not in N:
                        N.append(edge.tgt)

        #print(".....")
        #print(N)

        for n in N:
            if n not in perm_mark:
                visit(n)
            
            if not stillGetAListEvenIfNotDag:
                if not dag:
                    return dag

        self.topo = L

        for s in self.topo:
            print(">", s)
        
        return dag



