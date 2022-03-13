#!/bin/bash
# Script to run jplag

DATA_DIR=/data
TMP_DATA_DIR=/data_tmp
RESULT_DIR=/result

if [ ! -d $TMP_DATA_DIR ]
then
    mkdir $TMP_DATA_DIR
fi

dirs=$(find "$DATA_DIR" -mindepth 1 -type d)
for DIR_CLASS in $dirs
do
    echo "Processing - $DIR_CLASS"
    files=$(find "$DIR_CLASS" -mindepth 1 -type f)
    for FILE in $files
    do
        cp $FILE $TMP_DATA_DIR/$(basename $DIR_CLASS)_$(basename $FILE)
    done
done

java -jar jplag-2.12.1-SNAPSHOT-jar-with-dependencies.jar -m 1% -l c/c++ $TMP_DATA_DIR > results.txt
rm -rf $RESULT_DIR
mkdir $RESULT_DIR
mv results.txt $RESULT_DIR
python postprocessing.py