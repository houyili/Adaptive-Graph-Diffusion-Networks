cd "$(dirname $0)"
python -u ./main.py \
    --model agdn \
    --sample-type random_cluster \
    --train-partition-num 6 \
    --eval-partition-num 2 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 6 \
    --n-layers 6 \
    --weight-style HC \
    --dropout 0.4 \
    --n-hidden 150 \
    --input-drop 0.1 \
    --attn-drop 0. \
    --hop-attn-drop 0. \
    --edge-drop 0.1 \
    --norm none \
    --K 2 --edge-agg-mode "none_softmax" --edge-att-act="softplus" --norm="avg"\
    --gpu 2 --root "/home/lihouyi/ogb/gipa_plus/datasets/" --log-file-name="softplus_no_softmax_mean"
