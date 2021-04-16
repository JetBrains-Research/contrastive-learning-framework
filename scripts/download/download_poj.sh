#!/bin/bash
# Script - download codeforces dataset from s3
# options:
# $1              specify a percentage of dataset used as train set
# $2              specify a percentage of dataset used as test set
# $3              specify a percentage of dataset used as validation set
# $4              specify if developer mode is on, default: false
# $5              specify a path to astminer .jar file
# $6              specify a path to splitiing script
# $7              specify if splitted dataset needs to be downloaded

TRAIN_SPLIT_PART=$1
VAL_SPLIT_PART=$2
TEST_SPLIT_PART=$3
DEV=$4
ASTMINER_PATH=$5
SPLIT_SCRIPT=$6
LOAD_SPLITTED=$7
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
  if $LOAD_SPLITTED
  then
    if [ ! -f "$DATA_DIR/poj-104-splitted.tar.gz" ]
    then
      echo "Downloading splitted dataset ${DATASET_NAME}"
      wget https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/poj-104/poj-104-splitted.tar.gz -P $DATA_DIR/
    fi

    echo "Unzip splitted dataset"
    tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-splitted.tar.gz"
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
      tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz" "ProgramData/[1-3]"

      for i in {50..5000}
      do
        if [ -f "$DATA_DIR"/ProgramData/1/"$i".txt ]
        then
          rm "$DATA_DIR"/ProgramData/1/"$i".txt
        fi

        if [ -f "$DATA_DIR"/ProgramData/2/"$i".txt ]
        then
          rm "$DATA_DIR"/ProgramData/2/"$i".txt
        fi

        if [ -f "$DATA_DIR"/ProgramData/3/"$i".txt ]
        then
          rm "$DATA_DIR"/ProgramData/3/"$i".txt
        fi
      done
    else
      tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz"
    fi
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
  sh "$SPLIT_SCRIPT" "$DATA_PATH" "$DATA_PATH"_split "$TRAIN_SPLIT_PART" "$TEST_SPLIT_PART" "$VAL_SPLIT_PART"
  rm -rf "$DATA_PATH"
  mkdir "$DATA_PATH"
  mkdir "$DATA_PATH"/raw
  mv "$DATA_PATH"_split/* "$DATA_PATH"/raw
  rm -rf "$DATA_PATH"_split
fi

echo "Extracting paths using astminer. You need to specify the path to .jar in \"ASTMINER_PATH\" variable first"
if [ -d "$DATA_PATH"_parsed ]
then
  rm -rf "$DATA_PATH"_parsed
fi
mkdir "$DATA_PATH"_parsed

java -jar -Xmx200g $ASTMINER_PATH code2vec --lang c,cpp --project "$DATA_PATH"/raw/train --output "$DATA_PATH"_parsed/train --maxL 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx200g $ASTMINER_PATH code2vec --lang c,cpp --project "$DATA_PATH"/raw/test --output "$DATA_PATH"_parsed/test --maxL 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx200g $ASTMINER_PATH code2vec --lang c,cpp --project "$DATA_PATH"/raw/val --output "$DATA_PATH"_parsed/val --maxL 8 --maxW 2 --granularity file --folder-label --split-tokens

for folder in $(find "$DATA_PATH"_parsed/*/c -type d)
do
  for file in "$folder"/*
  do
    type="$(basename -s .csv "$(dirname "$folder")")"
    mv "$file" "$DATA_PATH"/"$(basename "${file%.csv}.$type.csv")"
  done
  rm -rf "$(dirname "$folder")"
done

rm -rf "$DATA_PATH"_parsed
