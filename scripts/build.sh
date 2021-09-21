#!/bin/bash
# Script - installing all the necessary libs

BUILD_DIR=build
CI=false

if [ ! -d "$BUILD_DIR" ]
then
  mkdir $BUILD_DIR
fi

while (( "$#" )); do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "--ci                pass it if build runs on CI"
      exit 0
      ;;
    --ci*)
      CI=true
      shift
      ;;
    *)
      echo "something went wrong"
      exit 1
  esac
done

if $CI
then
    VERSION=cpu
else
    VERSION=cu111
fi
echo $VERSION $CI
bash scripts/install_torch_geometric.sh $VERSION
bash scripts/install_astminer.sh
bash scripts/install_joern.sh
bash scripts/install_code_transformer.sh