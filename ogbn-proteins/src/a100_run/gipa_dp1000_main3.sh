cd "$(dirname $0)"
python -u ../main3.py \
    --model gipa_deep \
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
    --n-hidden 480 \
    --input-drop 0.1 \
    --norm none \
    --K 1 --edge-agg-mode "none_softmax" --edge-att-act="softplus"\
    --gpu 1 --root "/data/ogb/datasets/" --log-file-name="gipa_dp1000_main3"
