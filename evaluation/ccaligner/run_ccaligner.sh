#!/bin/bash

DIR_INPUT=/data/input
DIR_CLONES=/data/output

mkdir $DIR_CLONES

cd ccaligner
bash run_ccaligner.sh /data/input /data/output
mv /data/output/clones /data/output/clones.csv
