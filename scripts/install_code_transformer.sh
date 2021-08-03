#!/bin/bash

BUILD_DIR=build
CODE_TRANSFORMER=code-transformer
CONFIG_PATH=~/.config
CONFIGS_DIR="$CONFIG_PATH"/"$CODE_TRANSFORMER"
ENV_FILE_PATH="$CONFIGS_DIR"/.env

if [ ! -d "$CONFIG_PATH" ]
then
    mkdir "$CONFIG_PATH"
fi

if [ ! -d "$CONFIGS_DIR" ]
then
    mkdir "$CONFIGS_DIR"
fi

cp configs/code-transformer/.env "$ENV_FILE_PATH"

git clone https://github.com/maximzubkov/code-transformer.git "$BUILD_DIR"/"$CODE_TRANSFORMER"
cd "$BUILD_DIR"/"$CODE_TRANSFORMER" && git checkout cpp-dataset && pip install -e .
pip install git+https://github.com/tree-sitter/py-tree-sitter.git
