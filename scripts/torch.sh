 #! /bin/bash
 
 python -m experimart.training.train \
    -p torch/training.yml \
    -p wholeslidedata/dataiterator.yml