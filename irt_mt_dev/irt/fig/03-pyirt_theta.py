import irt_mt_dev.utils as utils

data_old = utils.load_data()
systems = list(data_old[0]["score"].keys())
system_ability = [19.11979103088379, 4.5034565925598145, 13.134197235107422, -11.980157852172852, -6.430403232574463, -3.4837207794189453, -5.468674659729004, 6.617031097412109, -10.058060646057129, 23.840787887573242, 7.919622898101807, 7.952557563781738, 0.734220027923584, -4.476377964019775, 18.284391403198242]
systems_irt_05 = {sys: sys_v for sys, sys_v in zip(systems, system_ability)}
systems_irt_us = {"NLLB_MBR_BLEU": -1.0462702512741089, "ONLINE-Y": 2.127668857574463, "ONLINE-G": -0.4774863123893738, "ONLINE-W": -0.5346319675445557, "CUNI-GA": -0.1966594159603119, "CUNI-DocTransformer": -0.1929299682378769, "ONLINE-B": 1.3501096963882446, "Lan-BridgeMT": 1.4047882556915283, "GPT4-5shot": -0.8781664967536926, "ZengHuiMT": 0.01718195155262947, "ONLINE-M": -1.4876989126205444, "CUNI-Transformer": -0.7942383885383606, "GTCOM_Peter": 0.09801753610372543, "ONLINE-A": -0.0003927212383132428, "NLLB_Greedy": 0.6077605485916138}

systems_wmt = utils.get_sys_ordering(data_old, metric="score")

print(utils.get_ord_accuracy(systems_irt_05, systems_wmt))
print(utils.get_ord_accuracy(systems_irt_us, systems_wmt))