#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1

export target_list="QED_benchmark_v0 QED_benchmark_v1 QED_benchmark_v2_1 QED_benchmark_v2_2 QED_benchmark_v2_3 QED_benchmark_v2_4 QED_benchmark_v3_1  QED_benchmark_v3_2 QED_benchmark_v3_3 QED_benchmark_v3_4 QED_benchmark_v3_5"

for t in $target_list
do
    for (( i=1; i <= 5; ++i ))
    do
        echo $t
        ./$t
    done
done
