#! /usr/bin/env bash
#set -x

#sleep 30
SCRIPT_DIR=$(cd $(dirname $0); pwd)

cd $SCRIPT_DIR
source venv/bin/activate
nohup python detector.py $@ &