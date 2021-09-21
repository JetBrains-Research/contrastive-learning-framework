#!/bin/bash
# Script - installing torch_geometric, a framework for working with graph data in pytorch

TORCH=1.9.0
CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())')

if [ $CUDA_AVAILABLE == "True" ]
then
    VERSION=cu111
else
    VERSION=cpu
fi

echo "Installing ${TORCH}-${VERSION}"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install torch-geometric