#! /bin/bash
for i in {1..10}
do
    for j in CiteSeer PubMed
    do
        for z in [[],[],[]] [[32,16,4],[],[]]
        do
            python train.py --dataset $j --dgm_layers $z
        done
    done
done