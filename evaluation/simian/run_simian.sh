#!/bin/bash
# Script for running simian

DATA_DIR=/data
DATA_INPUT="$DATA_DIR"/"$1"
DATA_OUTPUT="$DATA_DIR"/"$2"

java -jar simian/bin/simian-2.5.10.jar -formatter=yaml "$DATA_INPUT" > "$DATA_OUTPUT"
sed -i '1,4d' "$DATA_OUTPUT"
