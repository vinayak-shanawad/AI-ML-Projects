#!/bin/bash
 
set [-eux]

echo "Installing tensorboard..."

pip install tensorboard

echo "Restarting Jupyter server..."
restart-jupyter-server