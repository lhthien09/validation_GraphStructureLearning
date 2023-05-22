#! /bin/bash

for i in Cora CiteSeer PubMed
do
    for j in 1 3 5 10 20 
    do
        python ./stats/stats.py --dataset $i --k $j --distance hyperbolic --dim 4
    done
done