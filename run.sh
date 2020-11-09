#!/bin/bash

truncate --size 0 out/log.out
echo -n $@
PID=$(sbatch --requeue --parsable "$@" | tail -n1)
echo "	$PID"
echo $PID > .last.pid
#tail -f out/test.out
