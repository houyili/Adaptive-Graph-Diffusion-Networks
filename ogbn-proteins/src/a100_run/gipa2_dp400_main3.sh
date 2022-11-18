cd "$(dirname $0)"
python -u ../main3.py \
    --model gipa_deep2 \
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
    --n-hidden 460 \
    --input-drop 0.1 \
    --norm none \
    --K 0 --edge-agg-mode "none_softmax" --edge-att-act="softplus"\
    --gpu 1 --root "/data/ogb/datasets/" --log-file-name="gipa2_dp400_main3"
