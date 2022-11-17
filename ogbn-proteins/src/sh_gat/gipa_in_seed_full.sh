cd "$(dirname $0)"
python -u ../main_gipa.py \
          --n-runs 10 \
          --gpu 3 \
          --lr 0.01 \
          --n-hidden 50 \
          --batch_rate 10 \
          --root "/data/ogb/datasets/"\
          --cpu-start-from 65 \
          --sample-no-limit \
          --sample-type in_seed_sample_full\
          --agg-batch-norm \
          --edge-att-act softplus \
          --n-heads 20 \
          --eval-batch-rate 3 \
          --edge-drop 0.0 \
          --dropout 0.4
