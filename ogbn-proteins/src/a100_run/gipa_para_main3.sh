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
    --n-layers 6 \
    --dropout 0.4 \
    --n-hidden 200 \
    --input-drop 0.1 \
    --norm none \
    --K 1 \
    --edge-agg-mode "none_softmax" \
    --edge-att-act="softplus"\
    --gpu 2 \
    --root "/data/ogb/datasets/"\
    --n-hidden-per-head 40 \
    --log-file-name="gipa_para_main3"
