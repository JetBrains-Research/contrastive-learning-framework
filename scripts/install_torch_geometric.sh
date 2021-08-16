#!/bin/bash
# Script - installing torch_geometric, a framework for working with graph data in pytorch

TORCH=1.7.0
CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())')

if [ $CUDA_AVAILABLE == "True" ]
then
    VERSION=cu102
else
    VERSION=cpu
fi

echo "Installing ${TORCH}-${VERSION}"

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${VERSION}.html
pip install --no-index torch-geometric