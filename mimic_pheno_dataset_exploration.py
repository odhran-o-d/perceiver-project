# %%
from z_cifar_npt_files.data_loaders import load_mimic_pheno

train_loader, val_loader, num_features, num_targets, time_code, structure = load_mimic_pheno(batch_size=32)



# %%
data = next(iter(train_loader))
# %%
