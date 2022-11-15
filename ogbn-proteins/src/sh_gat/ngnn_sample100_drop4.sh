cd "$(dirname $0)"
python -u ../main_gat_2.py --use-labels --n-runs 10 --gpu 3 --lr 0.008 --n-hidden 120 --batch_rate 10 \
--root "/data/ogb/datasets/" --cpu-start-from 40 --sample6 100 --sample5 100 --sample4 100 --dropout=0.4
