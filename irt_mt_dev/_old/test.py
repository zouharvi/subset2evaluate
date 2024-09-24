import irt_mt_dev.utils as utils
import numpy as np

data_old = utils.load_data()

# 'metric_SacreBLEU_bleu': 50.67309892897293, 'metric_SacreBLEU_chrf': 0.6685048422038385, 'metric_SacreBLEU_ter_neg': -0.5, 'metric_COMET': 0.810630738735199, 'metric_COMET_src': 0.00198473921045661, 'metric_Prism_ref': -1.248391628265381, 'metric_Prism_src': -3.505037784576416, 'metric_BERT_SCORE': 0.9561068415641785, 'metric_BLEURT_default': 0.5591212511062622, 'metric_BLEURT_large': 0.6688846945762634, 'metric_CharacTER_neg': -0.4268292682926829, 'metric_ESIM_': 0.8101189732551575}}
data_new = [line for line in data_old if np.average([sys_v["scores"]["metric_SacreBLEU_bleu"] for sys_v in line.values()]) < 100]
print(f"New prop: {len(data_new)/len(data_old):.1%}")
print(utils.eval_data_pairs(data_new, data_old))