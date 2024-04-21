#!/usr/bin/bash

mkdir -p data figures
cd data
git clone --depth 1 https://github.com/MicrosoftTranslator/ToShipOrNotToShip.git
cd ToShipOrNotToShip/
python3 -c 'import evaluation.tools; evaluation.tools.load_data(use_cache=True)'
mv data.pickle ../toship21.pkl


wget https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz
tar -xzf mt-metrics-eval-v2.tgz
rm mt-metrics-eval-v2.tgz