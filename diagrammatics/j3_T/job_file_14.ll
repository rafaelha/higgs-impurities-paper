# @ shell=/bin/bash
# Script for LoadLeveler job steps
# @ job_name = h-imp
# @ error  = error_14.err
# @ output = log_14.out
# @ job_type = MPICH
# @ node_usage = shared
# @ node = 1
# @ tasks_per_node = 28
# @ environment = COPY_ALL
# @ notification = never
# @ notify_user =
# @ class = 28core
# @ queue
# exit on error
set -e
python3 ../j3-farm.py 14 15 True
