# @ shell=/bin/bash
# Script for LoadLeveler job steps
# @ job_name = j3-integral
# @ error  = 0_error.err
# @ output = 0_log.out
# @ job_type = MPICH
# @ node_usage = shared
# @ node = 1
# @ tasks_per_node = 32
# @ environment = COPY_ALL
# @ notification = never
# @ notify_user =
# @ class = 32core
# @ queue
# exit on error
set -e
python3  optical-conductivity-analytical.py
