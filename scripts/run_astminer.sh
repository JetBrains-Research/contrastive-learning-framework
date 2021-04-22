#!/bin/bash

DATA_DIR=data
DATASET_NAME=poj_104
ASTMINER_SOURCE=build/astminer
ASTMINER_BINARY=build/shadow/lib-0.6.jar
ASTMINER_PATH=${ASTMINER_SOURCE}/${ASTMINER_BINARY}
DATA_PATH=${DATA_DIR}/${DATASET_NAME}

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help                     show brief help"
      echo "-d, --dataset NAME             specify dataset name, available: codeforces, poj_104"
      echo "--data_dir PATH                the path to dir where the data is stored"
      echo "--astminer PATH                the path to astminer"
      exit 0
      ;;
    -d|--dataset*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        DATASET_NAME=$2
        shift 2
      else
        echo "Specify dataset name"
        exit 1
      fi
      ;;
    --data_dir*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        DATA_DIR=$2
        shift 2
      else
        echo "Specify data_dir path"
        exit 1
      fi
      ;;
    --astminer*)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        ASTMINER_SOURCE=$2
        shift 2
      else
        echo "Specify astminer path"
        exit 1
      fi
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

echo "Extracting paths using astminer. You need to specify the path to .jar in \"ASTMINER_PATH\" variable first"
if [ -d "$DATA_PATH"_parsed ]
then
  rm -rf "$DATA_PATH"_parsed
fi

if [ "$DATASET_NAME" == "poj_104" ]
then
  lang=c
elif [ "$DATASET_NAME" == "codeforces" ]
then
  lang=cpp
else
  echo "Dataset $DATASET_NAME does not exist"
fi

mkdir "$DATA_PATH"_parsed
java -jar -Xmx200g "$ASTMINER_PATH" code2vec --lang "$lang" --project "$DATA_PATH"/raw/train --output "$DATA_PATH"_parsed/train --maxL 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx200g "$ASTMINER_PATH" code2vec --lang "$lang" --project "$DATA_PATH"/raw/test --output "$DATA_PATH"_parsed/test --maxL 8 --maxW 2 --granularity file --folder-label --split-tokens
java -jar -Xmx200g "$ASTMINER_PATH" code2vec --lang "$lang" --project "$DATA_PATH"/raw/val --output "$DATA_PATH"_parsed/val --maxL 8 --maxW 2 --granularity file --folder-label --split-tokens

for folder in $(find "$DATA_PATH"_parsed/*/"$lang" -type d)
do
  for file in "$folder"/*
  do
    type="$(basename -s .csv "$(dirname "$folder")")"
    mv "$file" "$DATA_PATH"/"$(basename "${file%.csv}.$type.csv")"
  done
  rm -rf "$(dirname "$folder")"
done

rm -rf "$DATA_PATH"_parsed