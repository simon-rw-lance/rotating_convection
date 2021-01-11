#! /bin/bash

# Script to merge raw data sets. Usage:
#     . direc/to/this/script/file_merge.sh direc/to/simulation
# The script assumes you have pointed towards the location of the
# snapshot, analysis, and run_parameters subfolders. The output
# will be a single .h5 file for each subfolder, named in the format
#     subfolder_simname.h5
# where simname is determined by the read simname command.
# WARNING: merge_single.py can create extremely large .h5 files.

echo Simulation name:
read simname

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python3 $DIR/merge.py "$1"snapshots --cleanup
python3 $DIR/merge.py "$1"analysis --cleanup
python3 $DIR/merge.py "$1"run_parameters --cleanup

python3 $DIR/merge_single.py $1 $simname
