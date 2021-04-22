#!/bin/bash

DATA_DIR=
ASTMINER_SOURCE=build/astminer
ASTMINER_BINARY="build/shadow/lib-0.*.jar"
ASTMINER_PATH=${ASTMINER_SOURCE}/${ASTMINER_BINARY}
DATA_PATH=${DATA_DIR}/${DATASET_NAME}

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

mkdir "$DATA_PATH"/paths

for folder in $(find "$DATA_PATH"_parsed/*/"$lang" -type d)
do
  for file in "$folder"/*
  do
    type="$(basename -s .csv "$(dirname "$folder")")"
    mv "$file" "$DATA_PATH"/paths/"$(basename "${file%.csv}.$type.csv")"
  done
  rm -rf "$(dirname "$folder")"
done

rm -rf "$DATA_PATH"_parsed