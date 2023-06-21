import sys

# probably not the best practice
sys.path.insert(1, sys.path[0]+'/wp')

import config
import wp.wp_inter
import wp.call_graph

def main(argv):

    # package directory and name
    package_dir = "/Users/ingkarat/Documents/GitHub/scikit-learn/sklearn"
    package_name = "sklearn"

    # class and function name. 
    class_name = "PCA" 
    function_name = "fit"

    # MISC setting
    config.PRINT_DEBUG = False
    config.PRINT_SOME_INFO = False
    config.PATH_SHORTENING = package_dir  # To shorten printing path
    config.IMPL_LIMIT = 20  # If a WP contain implications more than this number at any
                             # point in the analysis, remove this WP. (default = 200)                        
    config.PRINT_TO_TERMINAL = True  # Print result 
    config.WRITE_TXT_RESULT_TO_FILE = True  # Write text result (and some info) to file
    config.WRITE_PKL_TO_FILE = True  # Write .pkl result to file
    config.COMPACT_RESULT = True  # Only show WP in text result 

    
    # Default = run a PCA on fit function
    if not argv:
        wp.wp_inter.main(package_dir, package_name, class_name, function_name, "???")
    
    elif argv[0] == "wp":
        print(len(argv))
        if len(argv) != 3:
            print("usage: py main.py wp CLASS_NAME FUNCTION_NAME")
        else:
            class_name = argv[1]
            function_name = argv[2]
            wp.wp_inter.main(package_dir, package_name, class_name, function_name, "???")

    elif argv[0] == "cg": # call graph
        if len(argv) != 3:
            print("usage: py main.py cg CLASS_NAME FUNCTION_NAME")
        else:
            class_name = argv[1]
            function_name = argv[2]
            graph_analyzer = wp.call_graph.main(package_dir, class_name, function_name)
            print("operator_main_func:", graph_analyzer.main_func)
            print("operator_main_class:", graph_analyzer.main_class)
            print("A DAG: ",graph_analyzer.call_graph.isDAG())
            print("A DAG2 (NOT consider self-loop): ",graph_analyzer.call_graph.isDAG2())
            graph_analyzer.call_graph.printGraph()

    elif argv[0] == "XD":
        ...

    else:
        print("MAYBE HELP PAGE HERE")


if __name__ == "__main__":
    main(sys.argv[1:])
    print("=== END ===")

"""
TODO
- update requirement.txt
- those RQ exp
- sub-tool for CG


Note
- issues around adding > 1 edge in the call graph

- revisit comments in this file. They are from previous iteration

- Might want to skip some files and directories (testing suits, etc)
    - Set the crawler to ignore those files 
    in REF_crawler.py
        - get_function_map(...)

minor TODO
- FILTERED WP. IT HAS > XX IMPLICATIONS AT SOME POINT
    A filtered can have this phase within the formula
    fix it so that if this WP is filted, its WP is only this phase
- make COMPACT mode to remove TRUE WPs too
"""
