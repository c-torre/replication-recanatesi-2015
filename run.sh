#!/bin/sh

logs_dir="logs"
mkdir -p $logs_dir
truncate --size 0 $logs_dir/log.out
echo -n $@
PID=$(sbatch --requeue --parsable "$@" | tail -n1)
echo "	$PID"
echo $PID > .last_pid
