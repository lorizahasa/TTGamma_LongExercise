#!/usr/bin/env bash
tar -zxf ttgenv.tar.gz
source ttgenv/bin/activate

tar -zxf ttgamma.tar.gz

head runFullDataset.py
python runFullDataset.py $1 --condor --chunksize 100000 --workers 2

pwd
ls -lrth
