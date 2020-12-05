# @ shell=/bin/bash
# Script for LoadLeveler job steps
# @ job_name = j3-integral
# @ error  = 0_error_parallel.err
# @ output = 0_log_parallel.out
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
python3  optical-conductivity-analytical.py >> log.out
