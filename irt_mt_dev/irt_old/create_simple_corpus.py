from transformers import pipeline
from irt_mt_dev import utils

pipes = []


def pipeline_base(model):
    return pipeline("translation_en_to_de", model=model, num_beams=5)

def pipeline_bad(model):
    return pipeline("translation_en_to_de", model=model, num_beams=1, temperature=1.1, do_sample=True)

pipes.append(("opus_base", lambda: pipeline_base("Helsinki-NLP/opus-mt-en-de")))
pipes.append(("opus_bad", lambda: pipeline_bad("Helsinki-NLP/opus-mt-en-de")))
pipes.append(("meta_base", lambda: pipeline_base("facebook/wmt19-en-de")))
pipes.append(("meta_bad", lambda: pipeline_bad("facebook/wmt19-en-de")))

data_old = utils.load_data()

for model_name, model_pipe in pipes:
    pass