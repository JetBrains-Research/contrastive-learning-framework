#!/bin/bash
# Script - installing joern, a library for Code Property Graphs from code

wget https://github.com/ShiftLeftSecurity/joern/releases/latest/download/joern-install.sh
mv joern-install.sh build
chmod +x ./build/joern-install.sh
sh ./build/joern-install.sh --reinstall