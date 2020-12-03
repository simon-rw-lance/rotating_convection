"""
Perform after execution of the merge.py script to combine all .h5 data files
into a singular .h5 file.

WARNING: This script may produce an extremely large output file.
"""

import pathlib
from dedalus.tools import post

# direc is the folder in which the various data folders are located

#-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#
############################# These strings need editting #############################
#-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#
run_name = "r15300"
direc_folder= f"np-1/t1e5_EVP_check/new_17nov_check/{run_name}/raw_data/"
#-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#
#######################################################################################
#-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#   #-----#


subfolder = "snapshots"
direc = direc_folder + subfolder
print(direc_folder)
# name of the resulting output file
output_name = f"{subfolder}_{run_name}"

set_paths = list(pathlib.Path(direc).glob(subfolder + "_s*.h5"))
post.merge_sets(direc + "/" + output_name + ".h5", set_paths, cleanup=True)

# direc is the folder in which the various data folders are located
subfolder = "analysis"
direc = direc_folder + subfolder
# name of the resulting output file
output_name = f"{subfolder}_{run_name}"

set_paths = list(pathlib.Path(direc).glob(subfolder + "_s*.h5"))
post.merge_sets(direc + "/" + output_name + ".h5", set_paths, cleanup=True)

# direc is the folder in which the various data folders are located
subfolder = "run_parameters"
direc = direc_folder + subfolder
# name of the resulting output file
output_name = f"{subfolder}_{run_name}"

set_paths = list(pathlib.Path(direc).glob(subfolder + "_s*.h5"))
post.merge_sets(direc + "/" + output_name + ".h5", set_paths, cleanup=True)
