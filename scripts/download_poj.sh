#!/bin/bash
# Script - download poj_104 dataset from s3
# options:
# $1              specify if developer mode is on, default: false

DEV=false
DATA_DIR=./data
DATASET_NAME=poj_104

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "--dev                          pass it if developer mode should be used, default false"
      exit 0
      ;;
    --dev*)
      DEV=true
      shift
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

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
    tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz" "ProgramData/[1-3]"
  else
    tar -C $DATA_DIR/ -xvf "$DATA_DIR/poj-104-original.tar.gz"
  fi

  mv "$DATA_DIR"/ProgramData "$DATA_PATH"
fi

# To prepare our dataset for astminer we need to rename all .txt files to .c files
echo "Renaming files"
find "$DATA_PATH" -depth -name "*.txt" -type f -exec sh -c 'echo "$0" && mv "$0" "${0%.txt}.cpp"' {} \;
