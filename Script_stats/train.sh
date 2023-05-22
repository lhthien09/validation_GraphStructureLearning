#! /bin/bash
for i in {1..10}
do
    for j in Cora CiteSeer PubMed
    do
        for z in 1 3 5 10 20
        do
            python train.py --dataset $j -k $z
        done
    done
done