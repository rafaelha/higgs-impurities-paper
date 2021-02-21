# @ shell=/bin/bash
# Script for LoadLeveler job steps
# @ job_name = h-imp
# @ error  = error_3.err
# @ output = log_3.out
# @ job_type = MPICH
# @ node_usage = shared
# @ node = 1
# @ tasks_per_node = 64
# @ environment = COPY_ALL
# @ notification = never
# @ notify_user =
# @ class = 32core
# @ queue
# exit on error
set -e
python3 ../j3-farm-QP.py 3 3 True
