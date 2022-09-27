from ttnn.utils.cv_utils import DatasetMode


# ########## Classification/Regression Mask Modes  ##########
# #### Stochastic Label Masking ####
# On these labels, stochastic masking may take place at training time, vali-
# dation time, or at test time.
# DATA_MODE_TO_LABEL_BERT_MODE = {
#     DatasetMode.TRAIN: [DatasetMode.TRAIN],
#     DatasetMode.VAL: [DatasetMode.TRAIN],
#     DatasetMode.TEST: [DatasetMode.TRAIN, DatasetMode.VAL],
# }
DATA_MODE_TO_LABEL_BERT_MODE = {
    'train': ['train'],
    'val': ['train'],
    'test': ['train', 'val'],
}

# However, even when we do stochastic label masking, some labels will be
# masked out deterministically, to avoid information leaks.
DATA_MODE_TO_LABEL_BERT_FIXED = {
    'train': ['val', 'test'],
    'val': ['val', 'test'],
    'test': ['test'],
}
# ########## Production Setting ##########
# Our code might be problematic when masking out entries which do not
# currently exist. Since in the production setting not the entire matrix is
# passed, we need to adjust the entries for which we perform masking.
DATA_MODE_TO_LABEL_MASK_MODE = {
    DatasetMode.TRAIN: [DatasetMode.TRAIN],
    DatasetMode.VAL: [DatasetMode.TRAIN, DatasetMode.VAL],
    DatasetMode.TEST: [DatasetMode.TRAIN, DatasetMode.VAL, DatasetMode.TEST],
}

# ########## Row-Independent Inference Models ##########
# E.g. the predictions for each row with an IMAB model are unaffected by
# one another at val/test-time.
MODEL_TO_ROW_INDEPENDENT_INFERENCE_PROPERTY = {
    'IMAB': True,
    'SAB': False,
    'ISAB': False
}
