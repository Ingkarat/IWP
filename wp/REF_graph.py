import config

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
  
  def rpl(self, x):
    return x.replace(config.PATH_SHORTENING,"")

  def printGraph(self):
    print("\n Printing the graph: ")    
    for key in self.edges.keys():
      for edge in self.edges[key]:
        print("Call from ",self.rpl(edge.src)," to ",self.rpl(edge.tgt))    
  

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
