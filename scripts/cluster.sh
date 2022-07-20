#!/bin/sh
ARCH=$1

if [ $# -ne 1 ]
  then
    echo "Arguments error: <ARCH>"
    exit 1
fi

python -u examples/cluster.py -d pitts -a ${ARCH} -b 48 --width 640 --height 480