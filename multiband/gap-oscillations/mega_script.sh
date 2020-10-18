#!/bin/bash

MAX=319
for i in `seq 0 1 ${MAX}`
do
	FILENAME=job_file_${i}.ll
	echo "# @ shell=/bin/bash" > $FILENAME

	echo "# Script for LoadLeveler job steps" >> $FILENAME

	echo "# @ job_name = h-imp" >> $FILENAME
	echo "# @ error  = error_${i}.err" >> $FILENAME
	echo "# @ output = log_${i}.out" >> $FILENAME
	echo "# @ job_type = MPICH" >> $FILENAME
	echo "# @ node_usage = shared" >> $FILENAME

	echo "# @ node = 1" >> $FILENAME
	echo "# @ tasks_per_node = 1" >> $FILENAME
	echo "# @ environment = COPY_ALL" >> $FILENAME
	echo "# @ notification = never" >> $FILENAME
	echo "# @ notify_user =" >> $FILENAME
	echo "# @ class = 28core" >> $FILENAME
	echo "# @ queue" >> $FILENAME

	echo "# exit on error" >> $FILENAME
	echo "set -e" >> $FILENAME

	echo "export OMP_NUM_THREADS=1" >> $FILENAME

	echo "python3 higgs_mgb2.py ${i} ${MAX}" >> $FILENAME


	llsubmit $FILENAME >> log.out
done
