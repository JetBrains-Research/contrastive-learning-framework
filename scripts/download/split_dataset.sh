#!/bin/bash
# Script - split data between train and test
# Default values
# options:
# -h, --help      show brief help
# $1              specify a dataset name
# $2              specify a directory where dataset is located
# $3              specify a directory to store output in
# $4              specify a percentage of dataset used as train set
# $5              specify a percentage of dataset used as test set
# $6              specify a percentage of dataset used as validation set

SHUFFLE=false

DATASET_NAME=$1
ORIGINAL_DATASET_PATH=$2
SPLIT_DATASET_PATH=$3
TRAIN_SPLIT_PART=$4
TEST_SPLIT_PART=$5
VAL_SPLIT_PART=$6
COMPUTE_BOUNDS_SCRIPT=scripts/download/compute_bounds.py

DIR_TRAIN="${SPLIT_DATASET_PATH}/train"
DIR_VAL="${SPLIT_DATASET_PATH}/val"
DIR_TEST="${SPLIT_DATASET_PATH}/test"

echo "Train $TRAIN_SPLIT_PART % "
echo "Val $VAL_SPLIT_PART %"
echo "Test $TEST_SPLIT_PART %"
echo "Shuffle $SHUFFLE"
echo "Original dataset path: ${ORIGINAL_DATASET_PATH}"
echo "Train dataset path: ${DIR_TRAIN}"
echo "Val dataset path = ${DIR_VAL}"
echo "Test dataset path = ${DIR_TEST}"

echo ""
echo "Removing all data inside ${SPLIT_DATASET_PATH}"
rm -rf "$SPLIT_DATASET_PATH"
mkdir "$SPLIT_DATASET_PATH"

mkdir "$DIR_TRAIN"
mkdir "$DIR_VAL"
mkdir "$DIR_TEST"

find "$ORIGINAL_DATASET_PATH" -mindepth 1 -type d -exec cp -r {} "$DIR_TRAIN" \;

num_files=$(find "$DIR_TRAIN" -mindepth 1 -type d | wc -l)

if [ $DATASET_NAME == "codeforces" ]
then
    basenames=$(find "$DIR_TRAIN" -mindepth 1 -type d | \
              awk -F '_' '{print $(NF-5)$(NF-4), $0}' | \
              sort -k1 | \
              cut -d ' ' -f2-)
    bounds=$(python $COMPUTE_BOUNDS_SCRIPT "$DIR_TRAIN" "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART")
    train_bound=$(echo $bounds | cut -d " " -f1)
    test_bound=$(echo $bounds | cut -d " " -f2)
    echo $train_bound
elif [ $DATASET_NAME == "poj_104" ]
then
    basenames=$(find "$DIR_TRAIN" -mindepth 1 -type d)
    train_bound=$(expr $num_files \* $TRAIN_SPLIT_PART / 100)
    test_bound=$(expr $train_bound + $num_files \* $TEST_SPLIT_PART / 100)
else
    echo "Dataset $DATASET_NAME does not exist"
fi

counter=$(expr 0)

for DIR_CLASS in $basenames
do
    echo "Splitting class - $DIR_CLASS"

    counter=$(expr $counter + 1)
    if [[ $counter -gt $train_bound ]] && [[ $counter -le $test_bound ]];
    then
        mv "$DIR_CLASS" "$DIR_TEST"
    fi

    if [[ $counter -gt $test_bound ]];
    then
        mv "$DIR_CLASS" "$DIR_VAL"
    fi
done

echo "Done"
