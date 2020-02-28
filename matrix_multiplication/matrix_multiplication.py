#!/usr/bin/env python                                            
#
# matrix_multiplication ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import sys, os
import MatMulBench

sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp

Gstr_title = """
                 _        _                        _ _   _       _ _           _   _             
                | |      (_)                      | | | (_)     | (_)         | | (_)            
 _ __ ___   __ _| |_ _ __ ___  __  _ __ ___  _   _| | |_ _ _ __ | |_  ___ __ _| |_ _  ___  _ __  
| '_ ` _ \ / _` | __| '__| \ \/ / | '_ ` _ \| | | | | __| | '_ \| | |/ __/ _` | __| |/ _ \| '_ \ 
| | | | | | (_| | |_| |  | |>  <  | | | | | | |_| | | |_| | |_) | | | (_| (_| | |_| | (_) | | | |
|_| |_| |_|\__,_|\__|_|  |_/_/\_\ |_| |_| |_|\__,_|_|\__|_| .__/|_|_|\___\__,_|\__|_|\___/|_| |_|
                              ______                      | |                                    
                             |______|                     |_|                                    
"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the 
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       matrix_multiplication.py 

    SYNOPSIS

        python matrix_multiplication.py                                         \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python matrix_multiplication.py   \\
                                in    out

    DESCRIPTION

        `matrix_multiplication.py` ...

    ARGS

        [-h] [--help]
        If specified, show help message and exit.
        
        [--json]
        If specified, show json representation of app and exit.
        
        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.
        
        [--savejson <DIR>] 
        If specified, save json representation file to DIR and exit. 
        
        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.
        
        [--version]
        If specified, print version number and exit. 

"""


class matrix_multiplication(ChrisApp):
    """
    An app to ....
    """
    AUTHORS                 = 'FNNDSC (emslade@bu.edu, jeff0410@bu.edu, haoyangw@bu.edu, kefan29@bu.edu, yse@bu.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A ChRIS plugin app'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to test matrix multiplication on GPU'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument('-C','--COE',
                            dest = 'COEnumber',
                            type = int,
                            optional = True,
                            help = "assign COE parameter",
                            default = '128')
        self.add_argument('-t','--timeSpent',
                            dest = 'ElapseTime',
                            type = bool,
                            optional = True,
                            help = "elapse time",
                            default = 'True')

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())

        Matrix_Multiply = MatMulBench.MatMulBench(
            COEnumber= options.COEnumber, #args.COEnumber,
            ElapseTime= options.ElapseTime#args.ElapseTime
        )
        d_MatrixMultiply = Matrix_Multiply.Run()

        # has to be directed to the output directory
        if options.ElapseTime == 'True':
            f = open("output.txt","w+")
            f.write(d_MatrixMultiply)
            f.close()


    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)

        
# ENTRYPOINT
if __name__ == "__main__":
    chris_app = matrix_multiplication()
    chris_app.launch()
