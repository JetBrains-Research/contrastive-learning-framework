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

java -jar jplag-3.0.0-jar-with-dependencies.jar -n -1 -m 0.001 -l cpp $TMP_DATA_DIR
rm -rf $RESULT_DIR/*.html
rm -rf $RESULT_DIR/*.js
rm -rf $RESULT_DIR/*.gif
rm -rf $RESULT_DIR/*.png
mv $RESULT_DIR/matches_avg.csv $RESULT_DIR/results.csv