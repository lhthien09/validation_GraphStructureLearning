#! /bin/bash
for i in {1..10}
do
    for j in Cora CiteSeer PubMed
    do
        for k in gcn gat edgeconv
        do
            for l in [[],[],[]] [[32,16,4],[],[]]
            do 
                python train.py --dataset $j --gfun $k --dgm_layers $l
            done
        done
    done
done