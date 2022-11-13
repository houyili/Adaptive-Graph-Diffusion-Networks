cd "$(dirname $0)"
python -u ./main.py \
    --model agdn_ma \
    --sample-type random_cluster \
    --train-partition-num 6 \
    --eval-partition-num 2 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 50 \
    --n-layers 6 \
    --weight-style HC \
    --dropout 0.4 \
    --n-hidden 18 \
    --input-drop 0.1 \
    --attn-drop 0. \
    --hop-attn-drop 0. \
    --edge-drop 0.1 \
    --norm none \
    --K 1 --edge-agg-mode "none_softmax" --edge-att-act="softplus" --norm="avg"\
    --gpu 4 --root "/home/lihouyi/ogb/dataset/" --log-file-name="agdn_ma_softplus_k1_head50"
