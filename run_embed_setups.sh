
for delay in  0 10; do
    for dim in 32 64 100 ; do
        for clusters in 32 64 100 ; do
            for threshold in 0.8 0.9  ; do
                for patchwid in 6 8 10 ; do
                    python3 TrainHandler.py --datasize 3000 --embed-train-samples 10 --embed-dim $dim \
                    --embed-cluster $clusters --embed-pos-threshold $threshold --delay $delay
                done
            done
        done
    done
done