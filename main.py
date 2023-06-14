import sys

# probably not the best practice
sys.path.insert(1, sys.path[0]+'/wp')

import config
import wp.wp_inter


def main(argv):

    # package directory and name
    package_dir = "/Users/ingkarat/Documents/GitHub/scikit-learn/sklearn"
    package_name = "sklearn"

    # class and function name. 
    class_name = "PCA" 
    function_name = "fit"

    # MISC setting
    config.PRINT_DEBUG = False
    config.PRINT_SOME_INFO = True
    config.PATH_SHORTENING = package_dir  # TO shorten printing path
    config.IMPL_LIMIT = 200  # If a WP contain implications more than this number at any
                             # point in the analysis, remove this WP. (default = 200)


    wp.wp_inter.main(package_dir, package_name, class_name, function_name, "???")

if __name__ == "__main__":
    main(sys.argv[1:])
    print("=== END ===")

"""
Note
- TODO: issues around adding > 1 edge in the call graph

- revisit comments in this file. They are from previous iteration
"""
