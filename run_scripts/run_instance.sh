cd ../

python train_zi2zi_cls.py --norm InstanceNorm \
                          --gpu_ids 1 \
                          --blur 0.1 \
                          --tl 0 \
                          --blur 0 \
                          --neg_balance 1.3 \
                          --min_ns 6 \
                          --max_ns 12 