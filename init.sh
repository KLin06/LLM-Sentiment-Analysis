#!/bin/bash
# add sudo if on regular linux
apt update -y 
apt install -y lshw git 
curl -fsSL https://ollama.com/install.sh | sh
pip install ollama scikit-learn pandas
ollama serve &
sleep 3
ollama run llama3:8b