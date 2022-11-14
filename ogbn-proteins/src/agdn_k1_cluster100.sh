
cd "$(dirname $0)"
python -u ./mainc.py \
    --model agdn \
    --sample-type m_cluster \
    --train-partition-num 100 \
    --eval-partition-num 10 \
    --eval-times 2 \
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
    --K 1 \
    --gpu 4 --root "/data/ogb/datasets/" --log-file-name="k1_m_cluster100" --n-runs=5
