#!/bin/bash
 
set [-eux]

echo "Installing tensorboard..."

conda activate base

pip install tensorboard

conda deactivate

echo "Restarting Jupyter server..."
restart-jupyter-server