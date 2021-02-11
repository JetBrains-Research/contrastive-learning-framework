#!/bin/bash
# Script - installing yttm from JetnrainsResearch fork

BUILD_DIR=build
YTTM=YouTokenToMe

if [ ! -d "$BUILD_DIR" ]
then
  mkdir $BUILD_DIR
fi

git clone https://github.com/JetBrains-Research/YouTokenToMe.git $BUILD_DIR/"$YTTM"
pip install -r "$BUILD_DIR"/"$YTTM"/requirements.txt
pip install -e "$BUILD_DIR"/"$YTTM"