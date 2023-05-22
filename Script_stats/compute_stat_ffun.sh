##! /bin/bash

for data in Cora CiteSeer PubMed
do
    for f in gcn gat mlp 
    do 
        python ./stats/stats.py --ffun $f --dataset $data 
    done
done
