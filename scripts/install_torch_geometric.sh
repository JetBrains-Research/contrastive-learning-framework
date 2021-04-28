#!/bin/bash
# Script - installing torch_geometric, a framework for working with graph data in pytorch

TORCH=1.7.1
CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())')

if [ $CUDA_AVAILABLE == "True" ]
then
    CUDA=cu102
else
    CUDA=cpu
fi

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric