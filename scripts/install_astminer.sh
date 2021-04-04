#!/bin/bash
# Script - installing ASTMiner, a library for mining
# path-based representations of code and more

BUILD_DIR=build
ASTMINER=astminer

git clone https://github.com/JetBrains-Research/astminer.git $BUILD_DIR/"$ASTMINER"
cd $BUILD_DIR/"$ASTMINER" && ./gradlew shadowJar