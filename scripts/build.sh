#!/bin/bash
# Script - installing all the necessary libs

BUILD_DIR=build

if [ ! -d "$BUILD_DIR" ]
thenq
  mkdir $BUILD_DIR
fi

sh scripts/install_astminer.sh
sh scripts/install_yttm.sh