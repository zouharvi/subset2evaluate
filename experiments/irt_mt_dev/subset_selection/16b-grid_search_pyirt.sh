|
mkdir -p computed/pyirt_grid/

function run_grid_search {
    echo $1 > computed/pyirt_grid/$2.out;
    echo $1 > computed/pyirt_grid/$2.err;
    nohup python3 experiments/irt_mt_dev/subset_selection/16a-grid_search_pyirt.py "$1" \
        2>>computed/pyirt_grid/$2.err \
        1>>computed/pyirt_grid/$2.out \
    &
}

# 2pl, 3pl, multidim_2pl ("dims": 2/3/4)
# TODO: amortized_1pl, uses "hidden" and encodes the text?

run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '2pl', 'dropout': 0.5, 'priors': 'vague' }" 2pl
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '3pl', 'dropout': 0.5, 'priors': 'vague' }" 3pl
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.5, 'priors': 'vague' }" 4pl
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.5, 'priors': 'vague' }" 4pl_fixedpred
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.5, 'priors': 'hiearchical' }" 4pl_hiearchical
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '3pl', 'dropout': 0.5, 'priors': 'hiearchical' }" 3pl_hiearchical

# running
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': 'multidim_2pl', 'dims': 2, 'dropout': 0.5, 'priors': 'vague' }" 2pl_multi_d2
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': 'multidim_2pl', 'dims': 4, 'dropout': 0.5, 'priors': 'vague' }" 2pl_multi_d4
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': 'multidim_2pl', 'dims': 8, 'dropout': 0.5, 'priors': 'vague' }" 2pl_multi_d8

run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.00, 'priors': 'vague' }" 4pl_dropout_00
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.25, 'priors': 'vague' }" 4pl_dropout_25
run_grid_search "{ 'epochs': 1000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.75, 'priors': 'vague' }" 4pl_dropout_75

run_grid_search "{ 'epochs': 2000, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.5, 'priors': 'vague' }" 4pl_e2k
run_grid_search "{ 'epochs': 500, 'deterministic': true, 'model_type': '4pl', 'dropout': 0.5, 'priors': 'vague' }" 4pl_e500


python3 experiments/irt_mt_dev/subset_selection/16a-grid_search_pyirt.py "{ 'epochs': 1000, 'deterministic': true, 'model_type': '2pl', 'dropout': 0.5, 'priors': 'vague' }"

