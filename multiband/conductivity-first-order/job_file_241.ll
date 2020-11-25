# @ shell=/bin/bash
# Script for LoadLeveler job steps
# @ job_name = h-imp
# @ error  = error_241.err
# @ output = log_241.out
# @ job_type = MPICH
# @ node_usage = shared
# @ node = 1
# @ tasks_per_node = 1
# @ environment = COPY_ALL
# @ notification = never
# @ notify_user =
# @ class = 28core
# @ queue
# exit on error
set -e
export OMP_NUM_THREADS=1
python3 higgs_mgb2.py 241 363
