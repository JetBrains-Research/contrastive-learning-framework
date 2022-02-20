#!/bin/bash
# Script for running ccaligner

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

cd ccaligner
./run $TMP_DATA_DIR $RESULT_DIR

DIR_EXTRACT=./_extract
DIR_PARSE=./_parse
DIR_TOKEN=./token
DIR_DETECT=./_detect

mkdir $DIR_EXTRACT
mkdir $DIR_PARSE
mkdir $DIR_TOKEN
mkdir $DIR_DETECT

chmod +x txl/*.x extract parser tokenize detect co1

pushd lexical
make clean
make
popd


./extract ./txl c functions $TMP_DATA_DIR $DIR_EXTRACT 8

./parser $DIR_EXTRACT $DIR_PARSE 5

./tokenize "${DIR_PARSE}/function.file" $DIR_TOKEN ./ 8

./detect $DIR_TOKEN $DIR_DETECT "${DIR_PARSE}/function.file" 6 1 0.6

./co1 $DIR_DETECT $RESULT_DIR

mv $RESULT_DIR/clones $RESULT_DIR/clones.csv