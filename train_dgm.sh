#! /bin/bash
for i in {1..10}
do
    for j in Cora CiteSeer PubMed
    do
        for z in [[],[],[]]
        do
            python train.py --dataset $j --dgm_layers $z
        done
    done
done