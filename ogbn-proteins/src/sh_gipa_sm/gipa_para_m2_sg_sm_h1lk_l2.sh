cd "$(dirname $0)"
python -u ../main2.py \
    --model gipa_para \
    --sample-type random_cluster \
    --train-partition-num 6 \
    --eval-partition-num 2 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 20 \
    --n-layers 2 \
    --dropout 0.4 \
    --n-hidden 100 \
    --input-drop 0.1 \
    --attn-drop 0. \
    --hop-attn-drop 0. \
    --edge-drop 0. \
    --edge-agg-mode "single_softmax" \
    --edge-att-act "none" \
    --norm none \
    --gpu 3 \
    --root "/data/ogb/datasets/"\
    --n-runs 10 \
    --n-hidden-per-head 50 \
    --log-file-name="gipa_para_m2_sg_sm_h1lk_l2"
