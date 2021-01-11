"""
Perform after execution of the merge.py script to combine all .h5 data files
into a singular .h5 file.

WARNING: This script may produce an extremely large output file.
"""

import pathlib
import sys
from dedalus.tools import post

direc = sys.argv[1]
run_name = sys.argv[2]

for subfolder in ['snapshots', 'analysis', 'run_parameters']:
    # print(f"Creating file {direc}{subfolder}/{subfolder}_{run_name}.h5")
    set_paths = list(pathlib.Path(f"{direc}{subfolder}").glob(f"{subfolder}_s*.h5"))
    post.merge_sets(f"{direc}{subfolder}/{subfolder}_{run_name}.h5", set_paths, cleanup=True)
