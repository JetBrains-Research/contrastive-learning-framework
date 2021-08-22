#!/bin/bash
# Script - installing ASTMiner, a library for mining
# path-based representations of code and more

BUILD_DIR=build
ASTMINER=astminer
COMMIT_HASH=c1e35b0a3d4d4ee4f6f13ac3d52a876313ce2cf7

git clone https://github.com/JetBrains-Research/astminer.git $BUILD_DIR/"$ASTMINER"
cd $BUILD_DIR/"$ASTMINER" && git checkout "$COMMIT_HASH" && ./gradlew shadowJar