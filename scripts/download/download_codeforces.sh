#!/bin/bash
# Script - download codeforces dataset from s3
# options:
# $1              specify a percentage of dataset used as train set
# $2              specify a percentage of dataset used as test set
# $3              specify a percentage of dataset used as validation set
# $4              specify if developer mode is on, default: false
# $5              specify a path to astminer .jar file

TRAIN_SPLIT_PART=$1
VAL_SPLIT_PART=$2
TEST_SPLIT_PART=$3
DEV=$4
SPLIT_SCRIPT=$5
DATA_DIR=./data
DATASET_NAME=codeforces

DATA_PATH=${DATA_DIR}/${DATASET_NAME}

if [ ! -d $DATA_DIR ]
then
  mkdir $DATA_DIR
fi

echo "Downloading dataset codeforces"
if [ -d "$DATA_PATH" ]
then
  echo "$DATA_PATH exists."
else
  if [ ! -f "$DATA_PATH".zip ]
  then
    wget https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/codeforces-code-clone/codeforces_reduced.zip
    mv codeforces_reduced.zip "$DATA_PATH".zip
  fi

  echo "Unzip dataset"

  if $DEV
  then
    unzip -qq "$DATA_PATH".zip 'codeforces_reduced/*_13[0-1][0-9]_[E-Z]_*' -d $DATA_DIR
  else
    unzip "$DATA_PATH".zip -d $DATA_DIR
  fi
  mv "${DATA_DIR}"/codeforces_reduced "$DATA_PATH"

  # Splitting dataset on train/test/val parts
  echo "Splitting on train/test/val"
  sh "$SPLIT_SCRIPT" "$DATASET_NAME" "$DATA_PATH" "$DATA_PATH"_split "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART"
  rm -rf "$DATA_PATH"
  mkdir "$DATA_PATH"
  mkdir "$DATA_PATH"/raw
  mv "$DATA_PATH"_split/* "$DATA_PATH"/raw
  rm -rf "$DATA_PATH"_split
fi
