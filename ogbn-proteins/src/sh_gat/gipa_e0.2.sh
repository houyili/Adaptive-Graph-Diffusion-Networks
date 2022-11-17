cd "$(dirname $0)"
python -u ../main_gipa.py --use-labels --n-runs 10 --gpu 1 --lr 0.01 --n-hidden 50 --batch_rate 10 \
--root "/data/ogb/datasets/" --cpu-start-from 40 --sample-no-limit --edge-sample-rate 0.10 \
--agg-batch-norm --edge-att-act softplus --n-heads 20 --eval-batch-rate 6 --edge-drop 0.0 --dropout 0.4
