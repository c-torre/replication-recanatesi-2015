#!/bin/sh

mkdir -p debug
truncate --size 0 debug/log.out
echo -n $@
PID=$(sbatch --requeue --parsable "$@" | tail -n1)
echo "	$PID"
echo $PID > .last.pid
