##! /bin/bash

for data in Cora CiteSeer PubMed
do
    for dgm in [[], [], []] [[32,16,4],[],[]]
    do 
        for g in gcn gat edgeconv 
        do 
            python ./stats/stats.py --gfun $g --dataset $data --dgm_layers $dgm
        done
    done
done
