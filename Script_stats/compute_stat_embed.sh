#! /bin/bash

for i in Cora CiteSeer PubMed
do
    for j in 2 6 8
    do
        python ./stats/stats.py --dataset $i --dim $j
    done
done