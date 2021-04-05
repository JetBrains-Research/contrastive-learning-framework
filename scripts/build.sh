#!/bin/bash
# Script - installing all the necessary libs

BUILD_DIR=build

if [ ! -d "$BUILD_DIR" ]
then
  mkdir $BUILD_DIR
fi

sh scripts/install_astminer.sh
sh scripts/install_yttm.sh
sh scripts/install_joern.sh
sh scripts/install_torch_geometric.sh