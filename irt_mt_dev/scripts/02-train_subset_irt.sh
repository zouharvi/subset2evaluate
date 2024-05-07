mkdir computed/

for METRIC in "MetricX-23-c" "BLEU" "score"; do
for SUBSET in "0.2" "0.4" "0.6" "0.8" "1.0"; do
    echo "###" $METRIC $SUBSET;
    python3 irt_mt_dev/irt/train.py \
        --no-save \
        --metric $METRIC \
        --binarize \
        --train-size $SUBSET;
done;
done;