import wandb
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from ttnn.model.ttnn import TTNNModel
from ttnn.utils.encode_utils import get_torch_dtype
from ttnn.utils.train_utils import count_parameters, init_optimizer
from ttnn.utils.train_utils import get_sorted_params


def is_implemented_model(model_type):
    if model_type not in {'nested', 'flattened', 'hybrid',
                          'hybrid-custom', 'i-npt', 'h-npt-large',
                          'hybrid-inducing', 'iterative-inducing'}:
        raise NotImplementedError


def is_flattened(model_type):
    """
    Args:
        model_type: str, either 'nested', 'flattened', or 'hybrid'.
    Returns:
        bool, is the model flattened.
        """
    is_implemented_model(model_type)
    return model_type in {
        'flattened', 'i-npt', 'h-npt-large', 'iterative-inducing'}


def has_column_attention(model_type):
    """
    Args:
        model_type: str, either 'nested', 'flattened', or 'hybrid'.
    Returns:
        bool, does the model perform attention over the columns (true
            only for nested and hybrid attention).
        """
    is_implemented_model(model_type)
    return model_type in {
        'nested', 'hybrid', 'hybrid-custom', 'hybrid-inducing'}


def init_model_opt_scaler_from_dataset(dataset, c, device=None,
                                       supcon_target_col=None):
    return init_model_opt_scaler(
        c, metadata=dataset.metadata, device=device,
        supcon_target_col=supcon_target_col)


def init_model_opt_scaler(
        c, metadata, device=None, supcon_target_col=None):
    if device is None:
        device = c.exp_device

    model = TTNNModel(
        c, metadata=metadata, device=device,
        supcon_target_col=supcon_target_col)

    model_torch_dtype = get_torch_dtype(dtype_name=c.model_dtype)
    model = model.to(device=device).type(model_torch_dtype)
    print(f'Model has {count_parameters(model)} parameters,'
          f'batch size {c.exp_batch_size}.')
    # get_sorted_params(model)

    optimizer = init_optimizer(
        c=c, model_parameters=model.parameters(), device=device)
    print(f'Initialized "{c.exp_optimizer}" optimizer.')

    # Automatic Mixed Precision (AMP)
    # If c.model_amp is False, the GradScaler call becomes a no-op
    # so we can switch between default/mixed precision without if/else
    # statements.
    scaler = GradScaler(enabled=c.model_amp)
    if c.model_amp:
        print(f'Initialized gradient scaler for Automatic Mixed Precision.')

    return model, optimizer, scaler


def setup_ddp_model(model, c, device):
    if not c.exp_azure_sweep and device == 0:
        wandb.watch(model, log="all", log_freq=10)

    # Deal with uncertainties module issues
    if c.model_multitask_uncertainties:
        uncertainties = model.uncertainties.to(device=device)

    # Deal with image patcher issues
    if c.model_image_n_patches:
        image_patcher = model.image_patcher.to(device=device)

    print(f'DDP with bucket size of {c.mp_bucket_cap_mb} MB.')

    # If we are not using train augmentation, we must "find unused params"
    # to avoid synchronizing gradients on the features
    find_unused_params = (c.model_augmentation_bert_mask_prob['train'] == 0)

    if find_unused_params:
        print('Finding unused params in DDP.')

    # Wrap model
    model = DDP(
        model, device_ids=[device], bucket_cap_mb=c.mp_bucket_cap_mb,
        find_unused_parameters=find_unused_params)

    if c.model_multitask_uncertainties:
        model.uncertainties = uncertainties
    else:
        model.uncertainties = None

    if c.model_image_n_patches:
        model.image_patcher = image_patcher
    else:
        model.image_patcher = None

    return model
