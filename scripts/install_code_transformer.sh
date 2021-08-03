#!/bin/bash

CONFIG_PATH=~/.config
CONFIGS_DIR="$CONFIG_PATH"/code_transformer
ENV_FILE_PATH="$CONFIGS_DIR"/.env

if [ ! -d "$CONFIG_PATH" ]
then
    mkdir "$CONFIG_PATH"
fi

if [ ! -d "$CONFIGS_DIR" ]
then
    mkdir "$CONFIGS_DIR"
fi

cp configs/code-transformer.env "$ENV_FILE_PATH"
