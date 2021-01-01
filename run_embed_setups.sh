#!/bin/sh
for delay in 0 10; do
    for dim in 64 100 ; do
        for clusters in 64 100 ; do
            for threshold in 0.8 0.9  ; do
                for patchwid in 6 8 10 ; do
                    python3 TrainHandler.py --datasize 10000 --embed-train-samples 1000 --embed-dim $dim \
                    --embed-cluster $clusters --embed-pos-threshold $threshold --delay $delay --embed-patch-width $patchwid
                done
            done
        done
    done
done