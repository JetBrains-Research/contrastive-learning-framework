#!/bin/bash
# Script to compute embedding from
# trans_coder-cpp (TransCoder_model_1.pth) and trans_coder-java (TransCoder_model_2.pth)

PYTHON_BIN_PATH=/root/miniconda3/envs/env/bin/python

$PYTHON_BIN_PATH run_transcoder.py --model_path=TransCoder_model_1.pth
$PYTHON_BIN_PATH run_transcoder.py --model_path=TransCoder_model_2.pth