cd "$(dirname $0)"
python -u ../main_gipa.py --use-labels --n-runs 10 --gpu 4 --lr 0.008 --n-hidden 50 --batch_rate 10 \
--root "/data/ogb/datasets/" --cpu-start-from 65 --sample6 100 --sample5 100 --sample4 100 --dropout=0.4 \
--agg-batch-norm --edge-att-act softplus --n-heads 20
