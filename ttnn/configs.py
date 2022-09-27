"""Define argument parser."""
import argparse

DEFAULT_AUGMENTATION_BERT_MASK_PROB = {
    'train': 0.15,
    'val': 0.,
    'test': 0.
}
DEFAULT_LABEL_BERT_MASK_PROB = {
    'train': 1,
    'val': 1,
    'test': 1
}


def build_parser():
    """Build parser."""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    ###########################################################################
    # #### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument(
        '--project', type=str, default='ttnn-debug-odhran',
        help='Wandb project name.')
    parser.add_argument(
        '--wandb_entity', type=str, default='oatml_tab',
        help='Entity for wandb')
    parser.add_argument(
        '--wandb_dir', type=str, default='./',
        help='Directory to which wandb logs.')

    ###########################################################################
    # #### Data Config ########################################################
    ###########################################################################

    parser.add_argument(
        '--data_path', type=str, default='data',
        help='Path of data')
    parser.add_argument(
        '--data_set', type=str, default='breast-cancer',
        help=f'Currently supported are breast-cancer, connect-4, poker-hand, '
        f'breast-cancer-imputation, higgs, epsilon, criteo, forest-cover, '
        f'boston-housing, mnist, yacht, concrete, energy-efficiency, income, '
        f'sarcos, protein, protein-imputation')
    parser.add_argument(
        '--data_loader_nprocs', type=int, default=0,
        help='Number of processes to use in data loading. Specify -1 to use '
             'all CPUs for data loading. 0 (default) means only the main  '
             'process is used in data loading. Applies for serial and '
             'distributed training.')
    parser.add_argument(
        '--data_set_on_cuda', type='bool', default=False,
        help='Place the entire dataset and metadata necessary per epoch on '
             'the CUDA device. Appropriate for smaller datasets.')
    parser.add_argument(
        '--data_force_reload', default=False, type='bool',
        help='If True, reload CV splits, ignoring cached files.')
    parser.add_argument(
        '--data_log_mem_usage', default=False, action='store_true',
        help='If True, report mem size of full dataset, as loaded from cache.')
    parser.add_argument(
        '--data_clear_tmp_files', type='bool', default=False,
        help=f'If True, deletes all downloaded/unzipped files in the dataset '
             f'folder while the CV split is being materialized and cached '
             f'(e.g. necessary to keep Higgs footprint under 30GB on Azure).')
    parser.add_argument(
        '--data_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                         'used for data (e.g. ground truth, masked inputs).')

    ###########################################################################
    # #### Experiment Config ##################################################
    ###########################################################################
    parser.add_argument(
        '--exp_device', default=None, type=str,
        help='If provided, use this (CUDA) device for the run.')
    parser.add_argument(
        '--exp_azure_sweep', default=False, type='bool',
        help='If True, delete logs from server and do not wandb.watch the '
             'model.')
    parser.add_argument(
        '--exp_test_equivariance', default=False, action='store_true',
        help='If True, run an equivariance test for the given model.')
    parser.add_argument(
        '--exp_show_empirical_label_dist', default=False, action='store_true',
        help='If True, reports the empirical dist over labels for each split.')
    parser.add_argument(
        '--exp_smoke_test', default=False, action='store_true',
        help='Run smoke test -- simpler data, check the pipes.')
    parser.add_argument(
        '--exp_name', type=str, default=None,
        help='Give experiment a name.')
    parser.add_argument(
        '--exp_group', type=str, default=None,
        help='Give experiment a group name.')
    parser.add_argument(
        '--np_seed', type=int, default=42,
        help='Random seed for numpy. Set to -1 to choose at random.')
    parser.add_argument(
        '--torch_seed', type=int, default=42,
        help='Random seed for torch. Set to -1 to choose at random.')
    parser.add_argument(
        '--baseline_seed', type=int, default=42,
        help='Random seed for torch. Set to -1 to choose at random.')
    parser.add_argument(
        '--exp_disable_cuda', dest='exp_use_cuda', default=True,
        action='store_false', help='Disable GPU acceleration')
    parser.add_argument(
        '--exp_n_runs', type=int, default=1,
        help=f'Maximum number of CV runs. This upper bounds the number of '
             f'cross validation folds that are being executed.')
    parser.add_argument(
        '--exp_batch_size', type=int, default=-1,
        help='Number of instances (rows) in each batch '
             'taken as input by the model. -1 corresponds to no '
             'minibatching.')
    parser.add_argument(
        '--exp_batch_mode_balancing', type='bool', default=True,
        help='Maintain relative train, val, and test proportions in batches.')
    parser.add_argument(
        '--exp_batch_class_balancing', type='bool', default=False,
        help='Maintain relative class proportions amongst train rows in '
             'batches.')
    parser.add_argument(
        '--exp_full_batch_gd', dest='exp_minibatch_sgd',
        default=True, action='store_false',
        help='Full batch gradient descent (as opposed to mini-batch GD)')
    parser.add_argument(
        '--exp_val_perc', type=float, default=0.1,
        help='Percent of examples in validation set')
    parser.add_argument(
        '--exp_test_perc', type=float, default=0.2,
        help='Percent of examples in test set. '
             'Determines number of CV splits.')
    parser.add_argument(
        '--exp_num_total_steps', type=float, default=100e3,
        help='Number of total gradient descent steps. The maximum number of '
             'epochs is computed as necessary using this value (e.g. in '
             'gradient syncing across data parallel replicates in distributed '
             'training).')
    parser.add_argument(
        '--exp_patience', type=int, default=-1,
        help='Early stopping -- number of epochs that '
             'validation may not improve before training stops. '
             'Turned off by default.')
    parser.add_argument(
        '--exp_checkpoint_setting', type=str, default='best_model',
        help='Checkpointing -- determines if there is no checkpointing '
             '(None), only the best model thus far is cached '
             '(best_model), or all models that improve on val loss should be '
             'maintained (all_checkpoints). See ttnn/utils/eval_utils.py.')
    parser.add_argument(
        '--exp_cache_cadence', type=int, default=1,
        help='Checkpointing -- we cache the model every `exp_cache_cadence` '
             'times that the validation loss improves since the last cache. '
             'Set this value to -1 to disable caching.')
    parser.add_argument(
        '--exp_load_from_checkpoint', default=False, type='bool',
        help='If True, attempt to load from checkpoint and continue training.')
    parser.add_argument(
        '--exp_data_loader_testing', dest='exp_data_loader_testing',
        default=False, action='store_true',
        help='Uses main function to initialize standard config '
             'and test torch loaders.')
    parser.add_argument(
        '--trial_run', dest='trial_run',
        default=False,
        action='store_true', help='Check configuration of args for exps.')
    parser.add_argument(
        '--exp_print_every_nth_forward', dest='exp_print_every_nth_forward',
        default=False, type=int,
        help='Print during mini-batch as well for large epochs.')
    parser.add_argument(
        '--exp_eval_every_n', type=int, default=5,
        help='Evaluate the model every n steps/epochs. (See below).')
    parser.add_argument(
        '--exp_eval_every_epoch_or_steps', type=str, default='epochs',
        help='Choose whether we eval every n "steps" or "epochs".')
    parser.add_argument(
        '--exp_eval_test_at_end_only',
        default=False,
        type='bool',
        help='Evaluate test error only in last step.')
    parser.add_argument(
        '--exp_artificial_missing',
        default=0,
        type=float,
        help=(
            'Percentage of artificially injected missing data into '
            'protein dataset.'))

    # Optimization
    # -------------
    parser.add_argument(
        '--exp_optimizer', type=str, default='lookahead_lamb',
        help='Model optimizer: see ttnn/optim.py for options.')
    parser.add_argument(
        '--exp_lookahead_update_cadence', type=int, default=6,
        help='The number of steps after which Lookahead will update its '
             'slow moving weights with a linear interpolation between the '
             'slow and fast moving weights.')
    parser.add_argument(
        '--exp_optimizer_warmup_proportion', type=float, default=0.7,
        help='The proportion of total steps over which we warmup.'
             'If this value is set to -1, we warmup for a fixed number of '
             'steps. Literature such as Evolved Transformer (So et al. 2019) '
             'warms up for 10K fixed steps, and decays for the rest. Can '
             'also be used in certain situations to determine tradeoff '
             'annealing, see exp_tradeoff_annealing_proportion below.')
    parser.add_argument(
        '--exp_optimizer_warmup_fixed_n_steps', type=int, default=10000,
        help='The number of steps over which we warm up. This is only used '
             'when exp_optimizer_warmup_proportion is set to -1. See above '
             'description.')
    parser.add_argument(
        '--exp_lr', type=float, default=1e-3,
        help='Learning rate')
    parser.add_argument(
        '--exp_scheduler', type=str, default='flat_and_anneal',
        help='Learning rate scheduler: see ttnn/optim.py for options.')
    parser.add_argument(
        '--exp_multistep_milestones', type=str, default=None,
        help='Epoch milestones for multistep LR scheduler.')
    # List can be passed as
    # --exp_multistep_milestones "list([10, 20, 50])"
    parser.add_argument(
        '--exp_gradient_clipping', type=float, default=1.,
        help='If > 0, clip gradients.')
    parser.add_argument(
        '--exp_weight_decay', type=float, default=0,
        help='Weight decay / L2 regularization penalty. Included in this '
             'section because it is set in the optimizer. '
             'HuggingFace default: 1e-5')
    parser.add_argument(
        '--exp_tradeoff', type=float, default=0.5,
        help='Tradeoff augmentation and label losses. If there is annealing '
             '(see below), this value specifies the maximum weight assigned '
             'to augmentation (i.e. exp_tradeoff = 1 will start by completely '
             'prioritizing augmentation, and gradually shift focus to labels.'
             'total_loss = tradeoff * aug_loss + (1 - tradeoff) * label_loss')
    parser.add_argument(
        '--exp_tradeoff_annealing', type=str, default='cosine',
        help='Specifies a scheduler for the tradeoff between augmentation '
             'and label losses. See ttnn/optim.py.')
    parser.add_argument(
        '--exp_tradeoff_annealing_proportion', type=float, default=1,
        help='The TradeoffAnnealer will take this proportion of the total '
             'number of steps to complete its annealing schedule. When this '
             'value is set to -1, we determine this proportion by the '
             'exp_optimizer_warmup_proportion argument. If that argument '
             'is not set, we default to annealing over the total '
             'number of steps (which can be explicitly set with value 1).')
    parser.add_argument(
        '--exp_conditional_evaluation', default=False, type='bool',
        help=(
            'If true, correctly factorise likelihood by conditioning on true '
            'values and predicting multiple times. '
            'Ignores row interactions.'))
    # Imputation
    # ------------
    # parser.add_argument(
    #     '--exp_imputation_val_entry_perc', type=float, default=0.25,
    #     help='Production imputation: percent of elements in '
    #          'validation rows that are held out')
    # parser.add_argument(
    #     '--exp_imputation_test_entry_perc', type=float, default=0.25,
    #     help='Production imputation: percent of elements in '
    #          'test rows that are held out')

    ###########################################################################
    # #### Multiprocess Config ################################################
    ###########################################################################

    parser.add_argument(
        '--mp_distributed', dest='mp_distributed', default=False, type='bool',
        help='If True, run data-parallel distributed training with Torch DDP.')
    parser.add_argument(
        '--mp_nodes', dest='mp_nodes', default=1, type=int,
        help='number of data loading workers')
    parser.add_argument(
        '--mp_gpus', dest='mp_gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument(
        '--mp_nr', dest='mp_nr', default=0, type=int,
        help='ranking within the nodes')
    parser.add_argument(
        '--mp_no_sync', dest='mp_no_sync', default=-1, type=int,
        help='Number of forward pass iterations for which gradients are not '
             'synchronized. Increasing this number will result in a lower '
             'amortized cost of distribution (and hence, a closer-to-linear '
             'scaling of per-epoch time with respect to number of GPUs), '
             'at the cost of convergence stability.')
    parser.add_argument(
        '--mp_bucket_cap_mb', dest='mp_bucket_cap_mb', default=25, type=int,
        help='Larger values denote a larger gradient bucketing size for DDP. '
             'This reduces the amortized overhead of communication, but means '
             'that there is a longer lead time before reduction (i.e. '
             'AllReduce aggregating gradient buckets across GPUs). Larger '
             'models (e.g. BERT ~ 110M parameters) will likely benefit '
             'from mp_bucket_cap_mb in excess of 50 MB.')

    ###########################################################################
    # #### Debug Config #######################################################
    ###########################################################################




    parser.add_argument(
        '--verbose', dest='verbose',
        default=False,
        action='store_true', help='If enabled, report train/val/test progress '
                                  'with tqdm.')

    parser.add_argument(
        '--debug_leakage', dest='debug_leakage',
        default=False,
        action='store_true', help=f'If enabled, flip all val/test labels.'
        f'Only works with breast cancer.')

    parser.add_argument(
        '--debug_label_leakage', dest='debug_label_leakage',
        default=False,
        type='bool', help=f'If enabled, print additional diagnostic '
                          f'information post-batching. Only works with '
                          f'breast-cancer and no stochastic label masking.')

    parser.add_argument(
        '--debug_row_interactions', dest='debug_row_interactions',
        default=False, type='bool',
        help=f'If enabled, performs a test of the capa'
        f'bilities of row interactions. '
        f'See scripts folder for execution of these debug settings.'
        )

    parser.add_argument(
        '--debug_eval_row_interactions', dest='debug_eval_row_interactions',
        default=False,
        type='bool',
        help='If enabled in a `duplicate` dataset setting '
             '(such as protein-duplicate), tests whether the model is '
             'successfully learning lookups by choosing a row to be '
             'evaluated, flipping the label of its duplicated row, and '
             'leaving all other duplicated rows alone.'
             'Alternatively, in a non-`duplicate` setting, we choose a row '
             'to be evaluated and independently permute each column among '
             'all other rows, to keep the batch statistics per column the '
             'same, but still destroy information that could be passed '
             'through row attention. '
             'If this is enabled, the main train loop is entered, but only '
             'for a model evaluation in all three dataset_modes.'
    )

    parser.add_argument(
        '--debug_eval_row_interactions_timer',
        dest='debug_eval_row_interactions_timer',
        default=None,
        type=float,
        help='Time allocated in hours to the debug_eval_row_interactions '
             'experiment. This flag is provided because evaluation of larger '
             'datasets can be intractable, as #rows forward passes are '
             'needed.'
    )


    parser.add_argument(
        '--debug_get_gpu_usage', dest='debug_get_gpu_usage',
        default=False, type='bool',
        help='If enabled will log GPU usage using the get_gpu_memory_map() function'
            'used only for the TTNN model'
             'Note: Functionality should be extend to log different features of the system'
        
    )

    parser.add_argument(
        '--debug_row_interactions_mode', dest='debug_row_interactions_mode',
        type=str, help=f'Different types of interaction debuging: '
        f'If MODE == dice-roll: '
        f'Target column is a dice roll. Need to copy dice row result from the '
        f'other rows. Need to set data_set=debug. '
        f'If MODE == crossword: '
        f'For each batch: List of words, + copy of that list with corrupted '
        f'letters. Only by matching across rows can we do this.'
        f'If MODE == lookup: '
        f'Like crossword, but only need to match a single column (lim. #rows).'
        f'If MODE == protein-duplicate: '
        f'Create unmasked duplicate of input dataset s.t. perfect performance '
        f'becomes possible by matching across rows.'
        f'If MODE == protein-duplicate-no-nn: '
        f'Create unmasked duplicate of input dataset s.t. perfect performance '
        f'becomes possible by matching across rows.'
        f'Additionally corrupt columns such that a nearest-neighbor based '
        f'matching is no longer succesful.'
        f'If MODE == protein-duplicate-no-nn-target-add: '
        f'Create unmasked duplicate of input dataset s.t. perfect performance '
        f'becomes possible by matching across rows.'
        f'Additionally corrupt columns such that a nearest-neighbor based '
        f'matching is no longer succesful.'
        f'Additionally add an offset to the target value. '
        f'This again will confuse k-NN but NPT should be able to learn it.'
    )

    parser.add_argument(
        '--debug_row_interactions_disable_after',
        type=int, default=-1, help=f'Disable debugging after n steps. '
        f'This is helpful e.g. if we use the protein-duplicate during '
        f'pretraining to bias the model towards learning row interactions.'
    )

    parser.add_argument(
        '--debug_no_stratify', dest='debug_no_stratify',
        default=False, action='store_true',
        help=f'Disable stratification in CV spltis for synthetic data.')

    parser.add_argument(
        '--debug_corrupt_standard_dataset_ablate_shuffle',
        dest='debug_corrupt_standard_dataset_ablate_shuffle',
        default=False, type='bool',
        help=f'Must be enabled with --debug_eval_row_interactions.'
             f'Disable the shuffling in the standard dataset corruption, but '
             f'still force the augmentation and label loss to only evaluate '
             f'one particular row. This will test our corruption mechanism -- '
             f'we should meet the performance without any corruption.'
    )

    parser.add_argument(
        '--debug_forward_cv_splits',
        dest='debug_forward_cv_splits',
        default=0, type=int,
        help=f'In case wandb runs have failed. Forward the cv splits to rescue'
             f'runs. Starting at 0, value gives the first cv split for which '
             f'we do a run.'
    )

    ###########################################################################
    # #### Model Class Config #################################################
    ###########################################################################

    parser.add_argument(
        '--model_class', dest='model_class', type=str,
        default='TTNN',  # in {TTNN, sklearn-baselines, DKL, ResNet, ...}
        help='Specifies model(s) to train/evaluate.')

    ###########################################################################
    # #### DKL Config #########################################################
    ###########################################################################
    parser.add_argument(
        '--dkl_hidden_layers',
        type=str, default=None,
        help='Hidden layers in feature extractor.')
    parser.add_argument(
        '--dkl_num_features',
        type=int, default=None,
        help='Num features output from feature extractor.')
    # List can be passed as
    # --dkl_hidden_layers "[100, 50, 10]"
    parser.add_argument(
        '--dkl_dropout_prob',
        type=float, default=None,
        help='Feature extractor dropout probability.')
    parser.add_argument(
        '--dkl_n_epochs',
        type=int, default=None)
    parser.add_argument(
        '--dkl_lr',
        type=float, default=None)
    parser.add_argument(
        '--dkl_n_inducing_points',
        type=int, default=None)
    parser.add_argument(
        '--dkl_enable_checkpointing',
        type='bool', default=True)
    parser.add_argument(
        '--dkl_patience',
        type=int, default=10)
    parser.add_argument(
        '--dkl_cv_split',
        type=int, default=None)

    ###########################################################################
    # #### scikit-learn Config ################################################
    ###########################################################################

    hyper_search_options = {'Bayes', 'Random', 'Grid'}
    parser.add_argument(
        '--sklearn_hyper_search', dest='sklearn_hyper_search', type=str,
        default='Grid',
        help=f'Specifies sklearn hyper search method;'
             f'in {hyper_search_options}.')
    parser.add_argument(
        '--sklearn_model', dest='sklearn_model', type=str,
        default='All',
        help=f'If specified, run tuning for the particular '
             f'sklearn model class (options: GradientBoosting, RandomForest, '
             f'CatBoost, SVM, MLP, XGBoost, TabNet, LightGBM. '
             f'New: Can also pass comma-separated list of '
             f'a selection of chosen models.')
    parser.add_argument(
        '--sklearn_verbose', dest='sklearn_verbose', type=int,
        default=1,
        help=f'Verbosity during hyper tuning. Higher number = more messages.'
             f' -1 for silent.')
    parser.add_argument(
        '--sklearn_n_jobs', dest='sklearn_n_jobs', type=int,
        default=-1,
        help=f'Number of cores to use during hyper tuning. '
             f'-1 is all available.')
    parser.add_argument(
        '--sklearn_val_final_fit', dest='sklearn_val_final_fit', type='bool',
        default=False,
        help=f'Use validation rows (in addition to training rows) in the final'
             f'fit of a baseline model.')

    # Imputation
    # ------------
    parser.add_argument(
        '--sklearn_impute_strategy', dest='sklearn_impute_strategy', type=str,
        default=None,
        help=f'Strategy used to perform sklearn imputation.'
             f'None implies that we allow the model to automatically '
             f'handle missing values. Mean and Median are '
             f'used in with the SimpleImputer, which simply uses univariate '
             f'column statistics. Other models, including BayesianRidge, '
             f'DecisionTreeRegressor, ExtraTreesRegressor, and '
             f'KNeighborsRegressor, XGBRegressor use round-robin style '
             f'imputation with IterativeImputer.')
    parser.add_argument(
        '--sklearn_iterative_imputer_max_iter',
        dest='sklearn_iterative_imputer_max_iter', type=int,
        default=10,
        help=f'Number of round-robin imputer iterations (where each iteration '
             f'imputes for all features once).')
    parser.add_argument(
        '--sklearn_iterative_imputer_imputation_order',
        dest='sklearn_iterative_imputer_imputation_order', type=str,
        default='ascending',
        help=f'The order in which the features will be imputed. '
             f'Possible values below.'
             f'ascending: From features with fewest missing values to most.'
             f'descending: From features with most missing values to fewest.'
             f'roman: Left to right.'
             f'arabic: Right to left.'
             f'random: A random order for each round.')

    ###########################################################################
    # #### General Model Config ###############################################
    ###########################################################################

    parser.add_argument(
        '--model_is_semi_supervised',
        default=True,
        type='bool', help='Include test features at training.')
    parser.add_argument(
        '--model_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                       'used for model.')
    parser.add_argument(
        '--model_amp',
        default=False,
        type='bool', help='If True, use automatic mixed precision (AMP), '
                          'which can provide significant speedups on V100/'
                          'A100 GPUs.')
    parser.add_argument(
        '--model_tfixup', type='bool', default=False,
        help='When True, use T-Fixup initialization (Huang et al. 2020), '
             'which allows for stable training of deep transformers with '
             'neither LayerNorm nor LR warmup. See ttnn/model/ttnn.py.'
             'http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf')
    parser.add_argument(
        '--model_feature_type_embedding', type='bool', default=True,
        help='When True, learn an embedding on whether each feature is '
             'numerical or categorical. Similar to the "token type" '
             'embeddings canonical in NLP. See https://github.com/huggingface/'
             'transformers/blob/master/src/transformers/models/bert/'
             'modeling_bert.py')
    parser.add_argument(
        '--model_feature_index_embedding', type='bool', default=True,
        help='When True, learn an embedding on the index of each feature '
             '(column). Similar to positional embeddings.')

    # #### Masking and Stochastic Forward Passes ##############################

    parser.add_argument(
        '--model_bert_augmentation', type='bool', default=True,
        help='When True, use BERT style augmentation. This introduces a mask '
             'column for each data entry, which can also be used to track '
             'values that are truly missing (i.e., for which there is no '
             'known ground truth). Set to False if: '
             '(i) You do not wish to use BERT augmentation'
             '(ii) You are confident there are no missing values in the '
             '      data.'
             '(iii) Given (i) and (ii), you do not want to include an '
             '      unneeded mask column for every entry.'
             'Note that you also must pass a model_augmentation_bert_mask_prob'
             ' dict with zeros for train, test, val.')
    parser.add_argument(
        '--model_bert_mask_percentage', type=float, default=0.9,
        help='Probability of actually masking out token after being selected.')
    parser.add_argument(
        '--model_augmentation_bert_mask_prob',
        type=str, default=DEFAULT_AUGMENTATION_BERT_MASK_PROB,
        help='Use bert style augmentation with the specified mask probs'
             'in training/validation/testing.')
    # Dicts can be passed as
    # --model_augmentation_bert_mask_prob "dict(train=.15, val=0, test=0)"
    parser.add_argument(
        '--model_num_eval_mc_samples', type=int, default=1,
        help='Number of stochastic forward passes at evaluation time, i.e. '
             'independent augmentation mask samples. We average predictions '
             'across these samples.')
    parser.add_argument(
        '--model_label_bert_mask_prob',
        type=str, default=DEFAULT_LABEL_BERT_MASK_PROB,
        help=f'Sample over the (train, train, train/val) labels at '
        f'(train, val, test) time "Labels" only exist in '
        f'classification/regression) setting, so this is where this mode'
        f'makes sense.')

    # #### (Supervised) Contrastive Regularization ############################

    parser.add_argument(
        '--model_supcon_regularizer', type='bool', default=False,
        help='Use the Supervised Contrastive Loss (SupCon) as a regularizer.'
             '(Khosla et al. 2020, https://arxiv.org/pdf/2004.11362.pdf)')
    parser.add_argument(
        '--model_supcon_feat_dim', type=int, default=128,
        help='Dimension of the Supervised Contrastive Learning normalized '
             'hidden vector, which is used in the computation of the loss. We '
             'project to this dimensionality from the TTNN hidden dimension '
             'output.')
    parser.add_argument(
        '--model_supcon_head', type=str, default='linear',
        help='Head used to project from TTNN hidden dimension to '
             'model_supcon_feat_dim.')
    parser.add_argument(
        '--model_supcon_temperature', type=float, default=0.1,
        help='')
    parser.add_argument(
        '--model_supcon_base_temperature', type=float, default=0.1,
        help='')
    parser.add_argument(
        '--model_supcon_contrast_mode', type=str, default='all',
        help='Contrast against `one` other example (SimCLR) or `all` other '
             'examples (SupCon). The prior setting depends on data '
             'augmentation and is pending implementation.')

    # #### Normalization ######################################################

    parser.add_argument(
        '--model_embedding_layer_norm', default=True, type='bool',
        help='(Disable) use of layer normalization after in-/out-embedding.')
    parser.add_argument(
        '--model_att_block_layer_norm', default=True, type='bool',
        help='(Disable) use of layer normalization in attention blocks.')
    parser.add_argument(
        '--model_layer_norm_eps', default=1e-12, type=float,
        help='The epsilon used by layer normalization layers.'
             'Default from BERT.')
    parser.add_argument(
        '--model_att_score_norm', default='softmax', type=str,
        help='Normalization to use for the attention scores. Options include'
             'softmax, constant (which divides by the sqrt of # of entries).')
    parser.add_argument(
        '--model_pre_layer_norm', default=True, type='bool',
        help='If True, we apply the LayerNorm (i) prior to Multi-Head '
             'Attention, (ii) before the row-wise feedforward networks. '
             'SetTransformer and Huggingface BERT opt for post-LN, in which '
             'LN is applied after the residual connections. See `On Layer '
             'Normalization in the Transformer Architecture (Xiong et al. '
             '2020, https://openreview.net/forum?id=B1x8anVFPr) for '
             'discussion.')

    # #### Dropout ############################################################

    parser.add_argument(
        '--model_hidden_dropout_prob', type=float, default=0.1,
        help='The dropout probability for all fully connected layers in the '
             '(in, but not out) embeddings, attention blocks.')
    parser.add_argument(
        '--model_att_score_dropout_prob', type=float, default=0.1,
        help='The dropout ratio for the attention scores.')

    ###########################################################################
    # #### Attention Block Config ############################################
    ###########################################################################

    parser.add_argument(
        '--model_type',
        default='hybrid',
        type=str,
        help='Choose model type. Current options: flattened, nested, hybrid,'
             'hybrid-custom, i-npt.')
    parser.add_argument(
        '--model_recurrence_interval',
        default=None,
        type=int,
        help=f'Experimental -- recurrently provide model with input through a ' 
             f'cross-attention layer, following Perceiver.')
    parser.add_argument(
        '--model_recurrence_share_weights',
        default=False,
        type='bool',
        help=f'If True, the same parameters are shared across the '
             f'`Core Blocks` and cross-attention layers in the recurrent '
             f'TTNN, like an RNN.')
    parser.add_argument(
        '--model_hybrid_debug',
        default=False,
        type='bool',
        help=f'Print dimensions of the input after each reshaping during '
             f'forward pass.')
    parser.add_argument(
        '--model_att_block',
        type=str,
        default='SAB',  # ['ISAB', 'SAB', 'IMAB']
        help=f'Specifies base attention block being used by the'
        f'Attention Between Datapoints (ABD) operation.' 
        f'If `model_type` \in [`flattened`], this will be the only'
        f'type of block used.')
    parser.add_argument(
        '--model_prototypes',
        type=int,
        default=0,
        help=f'Learn prototypical rows and concatenate to input.'
        f'Disabled by default')
    parser.add_argument(
        '--model_att_additive_encoding',
        type='bool',
        default=False,
        help=f'Apply additive encoding to the attention weights tensor '
        f'(pre-softmax) when performing nested attention over the columns.'
        f'Note that this argument has no effect for Flattened TTNN. '
        f'These allow the model to, for example, focus on modelling some '
        f'interactions between columns while avoiding others entirely at '
        f'a particular row. Inspired by tree-based methods.')
    parser.add_argument(
        '--model_att_multiplicative_encoding',
        type='bool',
        default=False,
        help=f'Apply multiplicative encoding to the attention scores tensor '
        f'(post-softmax) when performing nested attention over the columns.'
        f'Note that this argument has no effect for Flattened TTNN. '
        f'These allow the model to, for example, focus on modelling some '
        f'interactions between columns while avoiding others entirely at '
        f'a particular row. Inspired by tree-based methods.')
    parser.add_argument(
        '--model_nested_row_att_block',
        type=str,
        # ['ISAB', 'SAB', 'IMAB', 'IndepISAB', 'IndepSAB', 'IndepIMAB']
        default='SAB',
        help=f'If `model_type` \in [`nested`, `hybrid`], this argument will '
        f'set the AttentionBlock for the attention between atributes over rows N to be the '
        f'specified block. The AttentionBlock for attention over columns D is '
        f'still set by "model_att_block". '
        f'* Specifying IMAB in this argument may produce a transformer that '
             f'is compatible with mini-batching.'
        f'* Specifying one of the Indep blocks will learn a different '
             f'attention map for each column. This breaks equivariance w.r.t. '
             f'columns, which can be desirable.'
        f'* Note that currently, the number of '
             f'inducing points for IMAB/ISAB is just set by model_num_inds.')
    parser.add_argument(
        '--model_ablate_col_attention_only',
        type='bool',
        default=False,
        help=f'Only triggers if nested attention is enabled. This triggers an '
             f'ablation that replaces all row attention blocks with column '
             f'attention. How good is TTNN if we only use column attention?')
    parser.add_argument(
        '--model_ablate_col_attention_only_no_replace',
        type='bool',
        default=False,
        help=f'If model_ablate_col_attention_only=True, this option makes '
             f'sure that row att blocks are not replaced but simply left out.')

    parser.add_argument(
        '--model_ablate_rff',
        type='bool',
        default=False,
        help=f'Do not apply rFF during self-attention. Used to test the '
             f'hypothesis that the model does not learn row interactions '
             f'even in the flattened model because the FCNNs over the rows '
             f'are sufficiently expressive to solve the problem easily.')
    parser.add_argument(
        '--model_ablate_final_layer_rff',
        type='bool',
        default=False,
        help=f'Do not apply rFF during final layer.')
    parser.add_argument(
        '--model_use_pre_npt_rff',
        type='bool',
        default=False,
        help=f'Use a rFF to embed to hidden dimensions.')
    parser.add_argument(
        '--model_share_qk_sab_embedding',
        type='bool',
        default=False,
        help=f'Can only be enabled if only SAB blocks are exclusively used. '
             f'Will use the same embedding matrix for Queries and Keys. '
             f'Pushes us towards a type of attention that spreads '
             f'information between similar rows/columns.')
    parser.add_argument(
        '--model_qkv_embedding_identity_init',
        type='bool',
        default=False,
        help=f'If enabled, will initialize the Q, K, and V self-attention '
             f'embeddings centered around the identity. In other words, we '
             f'use an Xavier init centered around 1 for the main diagonal, '
             f'and an Xavier init centered around 0 elsewhere. '
             f'The intuition is that this will cause self-attention modules '
             f'to compute a KNN-like update, in which the most similar '
             f'tokens will update one another the most.'
    )
    parser.add_argument(
        '--model_checkpoint_key',
        type=str,
        default=None,
        help=f'If provided, use as title of checkpoint subdirectory. Used to '
             f'avoid collisions between subtly different runs.')

    ###########################################################################
    # #### Multihead Attention Config #########################################
    ###########################################################################

    parser.add_argument(
        '--model_dim_feat_embedding', type=int, default=16,
        help='Size of the input embedding per feature of the data.')
    # TODO: implement dim_target_embedding
    # parser.add_argument(
    #     '--model_dim_target_embedding', type=int, default=None,
    #     help='On occasion, the features can be easily embedded into a much '
    #          'smaller space than the target (e.g., binary features and a 100-'
    #          'class classification problem. In this case, use this feature to '
    #          'manually specify a higher `dim_target_embedding` for the target '
    #          'column(s). If unspecified, defaults to `dim_feat_embedding`.')
    parser.add_argument(
        '--model_dim_hidden', type=int, default=64,
        help='Intermediate feature dimension.')
    parser.add_argument(
        '--model_num_heads', type=int, default=8,
        help='Number of attention heads. Must evenly divide model_dim_hidden.')
    parser.add_argument(
        '--model_num_inds', type=int, default=16,
        help='Number of inducing points for ISAB or IMAB.')
    parser.add_argument(
        '--model_num_row_inds', type=int, default=-1,
        help='For nested TTNN: Option to use a different number of inducing '
             'points for between-rows attention. model_num_inds will hold the '
             'number of inducing points for between-columns attention.')
    parser.add_argument(
        '--model_sep_res_embed',
        default=True,
        type='bool',
        help='Use a seperate query embedding W^R to construct the residual '
        'connection '
        'W^R Q + MultiHead(Q, K, V)'
        'in the multi-head attention. This was not done by SetTransformers, '
        'which reused the query embedding matrix W^Q, '
        'but we feel that adding a separate W^R should only helpful.')
    parser.add_argument(
        '--model_stacking_depth',
        dest='model_stacking_depth',
        type=int,
        default=8,
        help=f'Number of layers to stack. Note that depth may not be directly '
        f'comparable between models. E.g. ISAB depth of 1 already contains '
        f'two MAB blocks.')
    parser.add_argument(
        '--model_mix_heads',
        dest='model_mix_heads',
        type='bool',
        default=True,
        help=f'Add linear mixing operation after concatenating the outputs '
        f'from each of the heads in multi-head attention.'
        f'Set Transformer does not do this for some reason. '
        f'We feel that this should only help. But it also does not really '
        f'matter as the rFF(X) can mix the columns of the multihead attention '
        f'output as well. '
        f'model_mix_heads=False may lead to inconsistencies for dimensions.')
    parser.add_argument(
        '--model_multitask_uncertainties',
        default=False,
        type='bool',
        help=f'Learn multitask uncertanties.')
    parser.add_argument(
        '--model_rff_depth',
        dest='model_rff_depth',
        type=int,
        default=1,
        help=f'Number of layers in rFF block.')
    parser.add_argument(
        '--model_rff_gated_gelu',
        dest='model_rff_gated_gelu',
        type='bool',
        default=False,
        help='If gated-gelu, use the HuggingFace implementation of the GeGLU '
            'rFF block, which has been found to improve performance across '
             'NLP tasks (https://arxiv.org/abs/2102.11972). Otherwise, build'
             'standard rFF block using GeLU activations and biases.')
    parser.add_argument(
        '--model_complex_in_embedding',
        default=False,
        type='bool',
        help=f'Use MLP to embed columns.')

    ###########################################################################
    # #### Image Patching  ####################################################
    ###########################################################################
    parser.add_argument(
        '--model_image_n_patches',
        default=False,
        type=int,
        help=f'If using TTNN for images, use image patching to handle high '
             f'per-sample dimensionality (and ultimately, attend over more'
             f' rows).')
    parser.add_argument(
        '--model_image_patch_type',
        default='linear',
        type=str,
        help='Determines the preprocessing applied per-patch. '
             'See model/ttnn.py.')
    parser.add_argument(
        '--model_image_n_channels',
        default=3,
        type=int,
        help='Number of channels (e.g. RGB = 3) in the input image.')
    parser.add_argument(
        '--model_image_share_embed',
        default=True,
        type='bool',
        help='If True, tie the embedding weights for each patch across '
             'the image (this is what Image Transformer does with its linear '
             'projection -- we also consider toggling this with convs and '
             'slot attention.')
    # TODO: remove this, @nband being lazy for a proof of concept
    parser.add_argument(
        '--model_image_n_classes',
        default=10,
        type=int,
        help='Number of dataset classes (e.g. MNIST = 10).')
    parser.add_argument(
        '--model_image_rand_augment',
        default=True,
        type='bool',
        help='Enables random augmentation.')
    parser.add_argument(
        '--model_image_random_crop_and_flip',
        default=True,
        type='bool',
        help='Enables random crop and flip.')
    parser.add_argument(
      '--model_image_rand_augment_n',
      default=3,
      type=int,
      help='RandAugment hyperparameter, from '
           'https://github.com/ildoonet/pytorch-randaugment/blob/master/'
           'confs/wresnet28x10_cifar10_b256.yaml')
    parser.add_argument(
      '--model_image_rand_augment_m',
      default=5,
      type=int,
      help='RandAugment hyperparameter, from '
           'https://github.com/ildoonet/pytorch-randaugment/blob/master/'
           'confs/wresnet28x10_cifar10_b256.yaml')
    parser.add_argument(
      '--model_image_rand_augment_cutout',
      default=16,
      type=int,
      help='RandAugment hyperparameter, from '
           'https://github.com/ildoonet/pytorch-randaugment/blob/master/'
           'confs/wresnet28x10_cifar10_b256.yaml')
    parser.add_argument(
      '--model_image_validity_test',
      default=False,
      type='bool',
      help='Compute a hash of the sum of the train, validation, and test '
           'sets to make sure it is actually the same.')
    parser.add_argument(
      '--image_test_patch_permutation',
      default=0,
      type=int,
      help='Randomly permute patches for every image. '
           '0 = do not do this; 1 = fixed permutation; 2 = random permutation')

    ###########################################################################
    # #### Metrics  ###########################################################
    ###########################################################################

    parser.add_argument(
        '--metrics_auroc',
        default=True,
        type='bool',
        help=f'Evaluate auroc metrics.'
        f'This option only has effect for classification datasets.'
        f'I.e. for datasets with target columns which are categorical.')

    ###########################################################################
    # #### Visualization  #####################################################
    ###########################################################################

    parser.add_argument(
        '--viz_att_maps',
        default=False,
        type='bool',
        help=f'Using config settings, attempt to load most recent checkpoint '
             f'and produce attention map visualizations.')
    parser.add_argument(
        '--viz_att_maps_save_path',
        default='data/attention_maps',
        type=str,
        help=f'Save attention maps to file. Specify the save path here.')

    ###########################################################################
    # #### Cluster dataset  ###################################################
    ###########################################################################

    parser.add_argument(
        '--cluster_N_samples',
        default=10,
        type=int,
        help=f'Number of samples for each cluster'
    )

    parser.add_argument(
        '--cluster_K_train',
        default=200,
        type=int,
        help=f'Number of cluster used for the training data'
    )

    parser.add_argument(
        '--cluster_K_test',
        default=20,
        type=int,
        help=f'Number of cluster used for the test data'
    )

    parser.add_argument(
        '--cluster_dimension',
        default=32,
        type=int,
        help=f'Dimension for the cluster encoding. This is equal to the number '
             f'of variables. This should be set high enough to distinguish '
             f'between the clusters.'
    )

    parser.add_argument(
        '--cluster_sigma',
        default=0.1,
        type=float,
        help=f'Sigma for label generation'
    )

    return parser


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")
