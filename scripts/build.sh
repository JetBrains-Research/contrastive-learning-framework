#!/bin/bash
# Script - installing all the necessary libs

BUILD_DIR=build

if [ ! -d "$BUILD_DIR" ]
then
  mkdir $BUILD_DIR
fi

bash scripts/install_astminer.sh
bash scripts/install_yttm.sh
bash scripts/install_joern.sh
bash scripts/install_torch_geometric.sh