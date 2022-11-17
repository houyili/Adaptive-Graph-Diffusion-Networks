cd "$(dirname $0)"
python -u ../main_gipa.py \
          --n-runs 10 \
          --gpu 4 \
          --lr 0.01 \
          --n-hidden 50 \
          --batch_rate 20 \
          --root "/data/ogb/datasets/"\
          --cpu-start-from 75 \
          --sample-no-limit \
          --sample-type in_seed_sample_full\
          --agg-batch-norm \
          --edge-att-act softplus \
          --n-heads 20 \
          --eval-batch-rate 3 \
          --edge-drop 0.0 \
          --dropout 0.4
