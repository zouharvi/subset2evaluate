for METRIC in "MetricX-23-c" "BLEU" "score"; do
for BINARIZE in "" "--binarize"; do
    echo "###" $METRIC $BINARIZE;
    python3 irt_mt_dev/irt/train.py --metric $METRIC $BINARIZE;
done;
done;