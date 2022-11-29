cd "$(dirname $0)"
python -u ../main4.py \
    --model gipa_sm \
    --sample-type random_cluster \
    --train-partition-num 6 \
    --eval-partition-num 2 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 20 \
    --n-layers 1 \
    --weight-style sum \
    --dropout 0.4 \
    --n-hidden 50 \
    --input-drop 0.1 \
    --attn-drop 0. \
    --hop-attn-drop 0. \
    --edge-drop 0.1 \
    --K 1 \
    --edge-agg-mode "single_softmax" \
    --edge-att-act "none" \
    --norm none \
    --edge-emb-size 16\
    --gpu 2 \
    --root "/data/ogb/datasets/" \
    --log-file-name="gipa_sm_single_softmax_h20_l1"
