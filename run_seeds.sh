#!/bin/bash
mkdir -p logs
for w in 2.0; do
  for s in 72 82; do
    python algorithms/IPPO/ippo_cnn_lstm_harvest_timeout.py SEED=$s ENV_KWARGS.intrinsic_motivation.im_w=$w > logs/svo_w${w}_seed_${s}.log 2>&1 || echo "w $w seed $s FAILED"
    echo "$(ls -td wandb/offline-run-* | head -1)  w=$w seed=$s" >> logs/run_index.txt
  done
done