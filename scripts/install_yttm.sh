#!/bin/bash
# Script - installing yttm from JetnrainsResearch fork

BUILD_DIR=build
YTTM=YouTokenToMe

git clone https://github.com/JetBrains-Research/YouTokenToMe.git $BUILD_DIR/"$YTTM"
pip install -r "$BUILD_DIR"/"$YTTM"/requirements.txt
pip install -e "$BUILD_DIR"/"$YTTM"