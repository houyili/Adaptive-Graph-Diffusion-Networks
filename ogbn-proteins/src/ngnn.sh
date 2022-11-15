cd "$(dirname $0)"
python -u ./main_gat_bot_ngnn.py --use-labels --n-runs 10 --gpu 0 --lr 0.008 --n-hidden 120 --batch_rate 10
      --root "/data/ogb/datasets/"
