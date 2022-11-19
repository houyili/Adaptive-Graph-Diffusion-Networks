cd "$(dirname $0)"
python -u ../main2.py \
    --model gipa_sm \
    --sample-type random_cluster \
    --train-partition-num 6 \
    --eval-partition-num 2 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 23 \
    --n-layers 6 \
    --weight-style sum \
    --dropout 0.4 \
    --n-hidden 20 \
    --input-drop 0.1 \
    --attn-drop 0. \
    --hop-attn-drop 0. \
    --edge-drop 0.0 \
    --K 1 \
    --edge-agg-mode "none_softmax" \
    --edge-att-act "softplus" \
    --norm none \
    --edge-emb-size 16\
    --gpu 4 \
    --root "/data/ogb/datasets/" \
    --seed 1 --n-runs 1\
    --log-file-name="gipa_sm_head23_seed1"
