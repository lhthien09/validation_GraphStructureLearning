#! /bin/bash
for i in {1..10}
do
    for j in Cora CiteSeer PubMed
    do
        for z in gcn gat mlp
        do
            python train.py --dataset $j --ffun $z
        done
    done
done