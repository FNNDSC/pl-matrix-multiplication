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

from __future__ import division
from numba import cuda, float32
import numpy
import math
import os
import sys
sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

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

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """

    @cuda.jit
    def fast_matmul(A, B, C):
        """
        Perform matrix multiplication of C = A * B
        Each thread computes one element of the result matrix C
        """

        # Define an array in the shared memory
        # The size and type of the arrays must be known at compile time
        sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

        x, y = cuda.grid(2)
        
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        if x >= C.shape[0] and y >= C.shape[1]:
            # Quit if (x, y) is outside of valid C boundary
            return

        # Each thread computes one element in the result matrix.
        # The dot product is chunked into dot products of TPB-long vectors.
        tmp = 0.
        for i in range(int(A.shape[1] / TPB)):
            # Preload data into shared memory
            sA[tx, ty] = A[x, ty + i * TPB]
            sB[tx, ty] = B[tx + i * TPB, y]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Computes partial product on the shared memory
            for j in range(TPB):
                tmp += sA[tx, j] * sB[j, ty]

            # Wait until all threads finish computing
            cuda.syncthreads()

        C[x, y] = tmp

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())
        fast_matmul(A, B, C)

        # The data array
        A = numpy.full((TPB*2, TPB*3), 3, numpy.float) # [32 x 48] matrix containing all 3's
        B = numpy.full((TPB*3, TPB*1), 4, numpy.float) # [48 x 16] matrix containing all 4's

        A_global_mem = cuda.to_device(A)
        B_global_mem = cuda.to_device(B)
        C_global_mem = cuda.device_array((TPB*2, TPB*1)) # [32 x 16] matrix result

        # Configure the blocks
        threadsperblock = (TPB, TPB)
        blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
        blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Start the kernel 
        fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        res = C_global_mem.copy_to_host()

        print(res)

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = matrix_multiplication()
    chris_app.launch()
