#!/bin/bash
# Script - download codeforces dataset from s3
# options:
# $1              specify a percentage of dataset used as train set
# $2              specify a percentage of dataset used as test set
# $3              specify a percentage of dataset used as validation set
# $4              specify if developer mode is on, default: false
# $5              specify a path to splitiing script

TRAIN_SPLIT_PART=$1
VAL_SPLIT_PART=$2
TEST_SPLIT_PART=$3
DEV=$4
SPLIT_SCRIPT=$5
DATA_DIR=./data
DATASET_NAME=poj_104

DATA_PATH=${DATA_DIR}/${DATASET_NAME}

if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

if [ -d "$DATA_PATH" ]
then
  echo "$DATA_PATH exists."
else
  if [ ! -f "$DATA_DIR/poj-104-original.tar.gz" ]
  then
    echo "Downloading dataset ${DATASET_NAME}"
    wget https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-original.tar.gz -P $DATA_DIR/
  fi

  echo "Unzip dataset"
  # In the developer mode we leave only several classes
  if $DEV
  then
    echo "Dev mode"
    tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz" "ProgramData/[1-6]"

    for i in {30..5000}
    do
      for j in {1..6}
      do
        if [ -f "$DATA_DIR"/ProgramData/"$j"/"$i".txt ]
        then
          rm "$DATA_DIR"/ProgramData/"$j"/"$i".txt
        fi
      done
    done
  else
    tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz"
  fi
  mv "$DATA_DIR"/ProgramData "$DATA_PATH"
fi

# To prepare our dataset for astminer we need to rename all .txt files to .c files
echo "Renaming files"
find "$DATA_PATH"/*/*  -name "*.txt" -type f -exec sh -c 'mv "$0" "${0%.txt}.c"' {} \;

if [ ! -d "$DATA_PATH"/raw ]
then
  # Splitting dataset on train/test/val parts
  echo "Splitting on train/test/val"
  sh "$SPLIT_SCRIPT" "$DATASET_NAME" "$DATA_PATH" "$DATA_PATH"_split "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART"
  rm -rf "$DATA_PATH"
  mkdir "$DATA_PATH"
  mkdir "$DATA_PATH"/raw
  mv "$DATA_PATH"_split/* "$DATA_PATH"/raw
  rm -rf "$DATA_PATH"_split
fi
