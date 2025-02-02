cd "$(dirname $0)"
python -u ../main3.py \
    --model gipa_para \
    --sample-type random_cluster \
    --train-partition-num 6 \
    --eval-partition-num 1 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 20 \
    --n-layers 5 \
    --dropout 0.4 \
    --n-hidden 60 \
    --input-drop 0.1 \
    --norm none \
    --K 1 \
    --edge-agg-mode "both_softmax" \
    --edge-att-act "none"\
    --gpu 5 \
    --root "/data/ogb/datasets/"\
    --n-hidden-per-head 47 \
    --log-file-name="gipa_para_softmax"
