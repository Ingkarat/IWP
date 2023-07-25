import sys
import os

# probably not the best practice
sys.path.insert(1, sys.path[0]+'/wp')

import config
import wp.wp_inter
import wp.call_graph
import experiment.json_printer

import pathlib

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
    config.IMPL_LIMIT = 10  # If a WP contain implications more than this number at any
                             # point in the analysis, remove this WP. (default = 200)                        
    config.PRINT_TO_TERMINAL = True  # Print result 
    config.WRITE_TXT_RESULT_TO_FILE = False  # Write text result (and some info) to file
    config.WRITE_PKL_TO_FILE = False  # Write .pkl result to file
    config.COMPACT_RESULT = True  # Only show WP in text result 

    current_path = str(pathlib.Path.cwd())

    def f_wp(package_dir, package_name, class_name, function_name, RQ_option):
        opAbsName = package_name + "_" + class_name + "_" + function_name
        text_path = current_path + "/output/" + package_name + "/text/[" + opAbsName + "]NoTrue"
        pkl_path = current_path + "/output/" + package_name + "/pkl/" + opAbsName + ".pkl"

        print(f"Starting weakest precondition analysis for {opAbsName}")
        wp.wp_inter.main(package_dir, package_name, class_name, function_name, RQ_option)
        print(f"End of weakest precondition analysis for {opAbsName}")
        print(f"> Text result is at {text_path}")
        print(f"> .pkl file is at {pkl_path}")

    def f_jss(package_dir, package_name, class_name, function_name):
        opAbsName = package_name + "_" + class_name + "_" + function_name
        pkl_file = pathlib.Path(current_path + "/output/" + package_name + "/pkl/" + opAbsName + ".pkl")
        output_path = current_path + "/output/" + package_name + "/jss/" + opAbsName + ".py"

        if pkl_file.is_file():
            pathlib.Path(current_path + "/output/" + package_name + "/jss").mkdir(parents=True, exist_ok=True)
            print(f"Starting JSON schema encoder for {opAbsName}")
            experiment.json_printer.to_json(str(pkl_file), output_path)
            print(f"End of JSON schema encoder for {opAbsName}")
            print(f"> Running 'black' for formatting.")
            os.system(f"black {output_path}")
            print(f"> JSON schemas is at {output_path}")
        else:
            print(f"No .pkl file for {opAbsName}. Please run the weakest precondition analysis first.")

    def rpl(x):
        return x.replace(config.PATH_SHORTENING,"")

    # Default = run a PCA on fit function
    if not argv:
        print(f"Default run of {class_name} class and {function_name} function.")
        f_wp(package_dir, package_name, class_name, function_name, "???")
    
    elif argv[0] == "wp":  # weakest precondition analysis
        print(len(argv))
        if len(argv) != 3:
            print("usage: py main.py wp CLASS_NAME FUNCTION_NAME")
        else:
            class_name = argv[1]
            function_name = argv[2]
            f_wp(package_dir, package_name, class_name, function_name, "???")

    elif argv[0] == "cg":  # call graph
        if len(argv) != 3:
            print("usage: py main.py cg CLASS_NAME FUNCTION_NAME")
        else:
            class_name = argv[1]
            function_name = argv[2]
            graph_analyzer = wp.call_graph.main(package_dir, class_name, function_name)
            print("operator_main_func:", rpl(graph_analyzer.main_func))
            print("operator_main_class:", rpl(graph_analyzer.main_class))
            print("A DAG: ",graph_analyzer.call_graph.isDAG())
            print("A DAG2 (NOT consider self-loop): ",graph_analyzer.call_graph.isDAG2())
            graph_analyzer.call_graph.printGraph()

    elif argv[0] == "jss": # json_printer
        if len(argv) != 3:
            print("usage: py main.py jss CLASS_NAME FUNCTION_NAME")
        else:
            class_name = argv[1]
            function_name = argv[2]
            f_jss(package_dir, package_name, class_name, function_name)

    elif argv[0] == "XD":
        ...

    else:
        print("MAYBE HELP PAGE HERE")


if __name__ == "__main__":
    main(sys.argv[1:])
    print("=== END ===")
