"""Contains main training operations."""

import gc
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from ttnn.column_encoding_dataset import ColumnEncodingDataset, TTNNDataset
from ttnn.loss import Loss
from ttnn.optim import LRScheduler
from ttnn.optim import TradeoffAnnealer
from ttnn.utils import debug
from ttnn.utils.batch_utils import collate_with_pre_batching
from ttnn.utils.encode_utils import torch_cast_to_dtype
from ttnn.utils.eval_checkpoint_utils import EarlyStopCounter, EarlyStopSignal
from ttnn.utils.logging_utils import Logger
from multiprocessing import cpu_count

class Trainer:
    def __init__(
            self, model, optimizer, scaler, c, wandb_run, cv_index,
            dataset: ColumnEncodingDataset = None,
            torch_dataset: TTNNDataset = None,
            distributed_args=None, supcon_target_col=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.c = c
        self.wandb_run = wandb_run
        self.is_distributed = False
        self.dataset = dataset
        self.torch_dataset = torch_dataset
        self.is_imputation = dataset.is_imputation
        self.supcon_target_col = supcon_target_col
        self.max_epochs = self.get_max_epochs()

        # Data Loading
        self.data_loader_nprocs = (
            cpu_count() if c.data_loader_nprocs == -1
            else c.data_loader_nprocs)

        if self.data_loader_nprocs > 0:
            print(
                f'Distributed data loading with {self.data_loader_nprocs} '
                f'processes.')

        # Only needs to be set in distributed setting; otherwise, submodules
        # such as Loss and EarlyStopCounter use c.exp_device for tensor ops.
        self.gpu = None

        if distributed_args is not None:
            print('Loaded in DistributedDataset.')
            self.is_distributed = True
            self.world_size = distributed_args['world_size']
            self.rank = distributed_args['rank']
            self.gpu = distributed_args['gpu']

        if c.exp_checkpoint_setting is None and c.exp_eval_test_at_end_only:
            raise Exception(
                'User is not checkpointing, but aims to evaluate the best '
                'performing model at the end of training. Please set '
                'exp_checkpoint_setting to "best_model" to do so.')

        self.early_stop_counter = EarlyStopCounter(
            c=c, data_cache_prefix=dataset.model_cache_path,
            metadata=dataset.metadata, wandb_run=wandb_run, cv_index=cv_index,
            n_splits=min(dataset.n_cv_splits, c.exp_n_runs),
            device=self.gpu)

        # Initialize from checkpoint, if available
        num_steps = 0

        if self.c.exp_load_from_checkpoint:
            checkpoint = self.early_stop_counter.get_most_recent_checkpoint()
            if checkpoint is not None:
                checkpoint_epoch, (
                    self.model, self.optimizer, self.scaler,
                    num_steps) = checkpoint

        self.scheduler = LRScheduler(
            c=c, name=c.exp_scheduler, optimizer=self.optimizer)

        # Initialize tradeoff annealer, fast forward to number of steps
        # recorded in checkpoint.
        if self.c.exp_tradeoff != -1 and not self.is_imputation:
            self.tradeoff_annealer = TradeoffAnnealer(
                c=c, num_steps=num_steps)
        else:
            self.tradeoff_annealer = None

        self.logger = Logger(
            self.c, self.optimizer, self.gpu, self.tradeoff_annealer)

        self.loss = Loss(
            self.c, self.model.uncertainties, dataset.metadata,
            device=self.gpu, tradeoff_annealer=self.tradeoff_annealer,
            is_minibatch_sgd=self.c.exp_minibatch_sgd)

        if self.c.exp_eval_every_epoch_or_steps == 'steps':
            self.last_eval = 0

    def init_new_dataset(self):
        dataset = ColumnEncodingDataset(self.c)
        if self.c.mp_distributed:
            raise NotImplementedError
        dataset.load_next_cv_split()
        self.dataset = dataset

    def get_distributed_dataloader(self, epoch):
        if not self.is_distributed:
            raise Exception

        sampler = torch.utils.data.distributed.DistributedSampler(
            self.torch_dataset,
            num_replicas=self.world_size,
            rank=self.rank)

        dataloader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=1,  # The dataset is already batched.
            shuffle=False,
            num_workers=self.data_loader_nprocs,
            pin_memory=True,
            collate_fn=collate_with_pre_batching,
            sampler=sampler)

        dataloader.sampler.set_epoch(epoch=epoch)
        total_steps = len(dataloader)

        if self.c.verbose:
            print('Successfully loaded distributed batch dataloader.')

        return dataloader, total_steps

    def get_num_steps_per_epoch(self):
        if self.c.exp_batch_size == -1:
            return 1

        N = self.dataset.metadata['N']
        return int(np.ceil(N / self.c.exp_batch_size))

    def get_max_epochs(self):
        # When evaluating row interactions:
        # We assume a trained model loaded from checkpoint.
        # Run two epochs:
        #   - (1) evaluate train/val/test loss without row corruptions
        #   - (2) evaluate train/val/test loss with row corruptions
        if self.c.debug_eval_row_interactions:
            return 2

        num_steps_per_epoch = self.get_num_steps_per_epoch()
        return int(
            np.ceil(self.c.exp_num_total_steps / num_steps_per_epoch))

    def per_epoch_train_eval(self, epoch):
        early_stop = False  # setting earl stop to false here
        if self.c.verbose:
            print(f'Epoch: {epoch}/{self.max_epochs}.')

        # need to increase step counter by one here (because step counter is)
        # still at last step
        end_experiment = (
                self.scheduler.num_steps + 1 >= self.c.exp_num_total_steps)

        # Immediately jump into end evaluation if we are debugging row interact
        end_experiment = end_experiment or (
            self.c.debug_eval_row_interactions and epoch == 2) #Odhran: check if forward pass with SAB - does the prediction for row A change when we change row B? Not used by default

        eval_model = end_experiment or self.eval_check(epoch)  # checking for a truth state

        if self.c.data_set == 'gmm':
            print('Sampling new GMM dataset.')
            self.init_new_dataset()  # Odhran - these are gaussian modes: will go through later

        if self.c.debug_eval_row_interactions: #this is a row corruption experiment to see if it effects eval mode. you would need to go through each row to permute which is super costly
            train_loss = None
        else:
            # The returned train loss is used for logging at eval time
            # It is None if minibatch_sgd is enabled, in which case we
            # perform an additional forward pass over all train entries
            train_loss = self.run_epoch(dataset_mode='train', epoch=epoch,
                                        eval_model=False)

        if eval_model:
            early_stop = self.eval_model(
                train_loss, epoch, end_experiment)
        if early_stop or end_experiment:
            early_stop = True
            return early_stop

        return early_stop

    def train_and_eval(self):
        ''''note to Odhran - main loop'''
        """Main training and evaluation loop."""
        self.logger.start_counting()

        if self.is_distributed and self.c.mp_no_sync != -1:
            curr_epoch = 1

            while curr_epoch <= self.max_epochs:
                with self.model.no_sync():
                    print(f'No DDP synchronization for the next '
                          f'{self.c.mp_no_sync} epochs.')

                    for epoch in range(
                            curr_epoch, curr_epoch + self.c.mp_no_sync):
                        if self.per_epoch_train_eval(epoch=epoch):
                            return

                        if epoch >= self.max_epochs:
                            return

                curr_epoch += self.c.mp_no_sync

                if epoch >= self.max_epochs:
                    return

                print(f'Synchronizing DDP gradients in this epoch '
                      f'(epoch {curr_epoch}).')
                if self.per_epoch_train_eval(epoch=curr_epoch):
                    return

                curr_epoch += 1

                '''this 'else' component is what gets handled if not distributed
                this is normally what gets handled'''
        else:
            for epoch in range(1, self.max_epochs + 1):
                if (self.c.data_set == 'forest-cover' and
                        self.c.debug_eval_row_interactions is True and
                        epoch == 1):
                    # Short circuit due to memory overflow on loading second
                    # model
                    continue
                if self.per_epoch_train_eval(epoch=epoch):
                    break

    def eval_model(self, train_loss, epoch, end_experiment):
        """Obtain val and test losses."""
        kwargs = dict(epoch=epoch, eval_model=True)

        # Evaluate over val rows
        val_loss = self.run_epoch(dataset_mode='val', **kwargs)

        if not (self.c.debug_eval_row_interactions and epoch == 2):  # again checking we're not doing that data corruption test.
            # Early stopping check -- TODO: consider loss other than label?
            counter, best_model_and_opt = self.early_stop_counter.update(
                val_loss=val_loss['label']['total_loss'],
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                end_experiment=end_experiment,
                tradeoff_annealer=self.tradeoff_annealer)
        else:
            counter = EarlyStopSignal.END

        if not self.c.debug_eval_row_interactions:
            if (counter == EarlyStopSignal.STOP) or end_experiment:
                if best_model_and_opt is not None:
                    print('Loaded best performing model for last evaluation.')  # loading in a previous model from checkpoint
                    self.model, self.optimizer, self.scaler, num_steps = (
                        best_model_and_opt)   # the scaler is used for mixed precision. not worth thinking about

                    # Initialize tradeoff annealer, fast forward to number of steps
                    # recorded in checkpoint.
                    if self.tradeoff_annealer is not None:
                        self.tradeoff_annealer = TradeoffAnnealer(
                            c=self.c, num_steps=num_steps)  # the loss from the features and the loss from the targets is switched over time. switch to target loss later


                        # Update the tradeoff annealer reference in the logger
                        self.logger.tradeoff_annealer = self.tradeoff_annealer

                # update val loss
                val_loss = self.run_epoch(dataset_mode='val', **kwargs) # eval test also uses run_epoch. this is the main meat!!

        if train_loss is None and not self.c.debug_eval_row_interactions:
            # Train and compute loss over masked features in train rows
            train_loss = self.run_epoch(dataset_mode='train', **kwargs)
        elif self.c.debug_eval_row_interactions:
            train_loss = {}

        # Check if we need to eval test
        if ((counter == EarlyStopSignal.STOP)
            or (not self.c.exp_eval_test_at_end_only)
                or (self.c.exp_eval_test_at_end_only and end_experiment)):
            # Evaluate over test and val rows again
            test_loss = self.run_epoch(dataset_mode='test', **kwargs)
        else:
            test_loss = None

        loss_dict = self.logger.log(
            train_loss, val_loss, test_loss, self.scheduler.num_steps, epoch) #Odhran: if you are going to log, you add logs here to the val loop

        # Update summary metrics
        new_min = (
            self.early_stop_counter.num_inc_valid_loss_epochs == 0)
        if (unc := self.model.uncertainties) is not None:
            self.logger.summary_log(loss_dict, new_min, unc.log())
        else:
            self.logger.summary_log(loss_dict, new_min)

        if counter == EarlyStopSignal.STOP:
            print(self.early_stop_counter.stop_signal_message)
            return True
        else:
            return False

    # fp = open('memory_profiler.log', 'w+')
    # @profile(stream=fp)
    # @profile
    def run_epoch(self, dataset_mode, epoch, eval_model=False):
        """Train or evaluate model for a full epoch.

        Args:
            dataset_mode (str) {'train', 'test', 'eval'}: Depending on value
                mask/input the relevant parts of the data.
            epoch (int): Only relevant for logging.
            eval_model (bool): If this is true, write some extra metrics into
                the loss_dict. Is always true for test and eval, but only
                sometimes true for train. (We do not log each train epoch).

        Returns:
            loss_dict: Results of model for logging purposes.

        If `self.c.exp_minibatch_sgd` is True, we backpropagate after every
        mini-batch. If it is False, we backpropagate once per epoch.
        """
        print_n = self.c.exp_print_every_nth_forward #Odhran: this is a helpful flag that shows you the loss at every forward pass. Good for massive datasets to show you are progressing.

        # Model prep
        # We also want to eval train loss
        if (dataset_mode == 'train') and not eval_model:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        # Dataset prep -- prepares dataset.batch_gen attribute
        # Relevant in 'production' setting: we want to only input train
        # at train, train/val at val and train/val/test at test.
        self.dataset.set_mode(mode=dataset_mode, epoch=epoch)

        # Initialize data loaders (serial / distributed, pinned memory)
        if self.is_distributed:
            # TODO: parallel DDP loading?
            self.torch_dataset.materialize(cv_dataset=self.dataset.cv_dataset)
            batch_iter, num_batches = self.get_distributed_dataloader(epoch)
        else:
            # TODO: can be beneficial to test > cpu_count() procs if our
            # loading is I/O bound (which it probably is)
            batch_dataset = self.dataset.cv_dataset #Odhran: this grabs the actual dataset
            extra_args = {}

            if not self.c.data_set_on_cuda:
                extra_args['pin_memory'] = True #Odhran: doesn't seem to matter too much - it's a way to prepare objects in CPU memory to prepare for transfer to GPU asynchronously.

            batch_iter = torch.utils.data.DataLoader(
                dataset=batch_dataset,
                batch_size=1,  # The dataset is already batched.
                shuffle=False,  # Already shuffled
                num_workers=self.data_loader_nprocs,  #Odhran: using multiple workers will duplicate batches in number of workers - could be a major bug
                collate_fn=collate_with_pre_batching,
                **extra_args)
            batch_iter = tqdm(
                batch_iter, desc='Batch') if self.c.verbose else batch_iter

        if (eval_model and self.c.debug_eval_row_interactions
                and epoch == 2 and dataset_mode in {'test'}):
            if self.c.debug_eval_row_interactions_timer is not None:
                self.set_row_corruption_timer()

        for batch_index, batch_dict_ in enumerate(batch_iter): #Odhran - this iterates through minibatches - if you want to debug or check stuff, this is where to do it.
            if self.c.image_test_patch_permutation != 0 and dataset_mode == 'train':
                continue
            # print(batch_dict)
            # debug: let's mess with val labels and see if it affects
            # debug: the train loss
            # for col in range(len(batch_dict['data_arrs'])):
            #     val_col_mask = batch_dict['val_mask_matrix'][:, col]
            #     if val_col_mask.sum() == 0:
            #         continue
            #     batch_dict['data_arrs'][col][val_col_mask] = torch.tensor(
            #         10000).double()
            if self.c.debug_row_interactions: #Odhran - this is an example of a good debug 
                batch_dict_ = debug.modify_data(
                    self.c, batch_dict_, dataset_mode,
                    self.scheduler.num_steps)

            # if True:
            #     a = 1

            # Perform a forward pass for each row (i.e. for a batch with N
            #   rows, perform N forward passes)
            # We do this on the second epoch in c.debug_eval_row_interactions
            # mode -- in the first epoch, we just do normal evaluation.
            # We also only do this for val and test, because it takes forever.
            # In each forward pass, we corrupt the other rows in a manner to
            # test if losing coherent row interactions will hurt performance
            # - For duplication experiments, we flip the label of the
            #       duplicate row of the chosen row_index
            # - For other experiments (e.g., standard datasets), we
            #       independently permute all other columns
            if (eval_model and self.c.debug_eval_row_interactions 
                    and epoch == 2 and dataset_mode in {'test'}):
                n_rows = batch_dict_['data_arrs'][0].shape[0]

                if (self.c.debug_row_interactions and
                    self.c.debug_row_interactions_mode == 'protein-duplicate'):
                    n_rows = n_rows // 2

                if batch_dict_['label_mask_matrix'] is not None:
                    # Triggers for stochastic label masking.
                    # Only not None in train mode. In which case this indicates the
                    # train labels that are masked, and will be evaluated on.
                    label_matrix_key = 'label'
                else:
                    # We are in val/test mode of stochastic label masking, or
                    # are just doing normal train/val/test with no stochastic
                    # label masking.
                    # In this case, the dataset_mode_mask_matrix tells us the
                    # location of all entries where we will compute a loss.
                    label_matrix_key = dataset_mode

                label_matrix = batch_dict_[f'{label_matrix_key}_mask_matrix']

                for row_index in range(n_rows):
                    # Only consider row_index where we would actually have
                    # been evaluating a loss.

                    # Note that for protein-duplication:
                    # we need to add the number of rows in the
                    # non-duplicated data, to actually have the appropriate
                    # index for the non-duplicated data.
                    if (self.c.debug_row_interactions and
                            self.c.debug_row_interactions_mode ==
                            'protein-duplicate'):
                        original_row_index = row_index + n_rows
                    else:
                        original_row_index = row_index

                    if label_matrix[original_row_index, :].sum() == 0:
                        continue

                    # This was only used when we were trying out the
                    # standard row corruption on a duplicated dataset.
                    # row_index = original_row_index

                    modified_batch_dict = debug.corrupt_rows(
                        self.c, batch_dict_, dataset_mode, row_index)

                    self.run_batch(modified_batch_dict, dataset_mode,
                                   eval_model, epoch, print_n, batch_index)

                if self.c.debug_eval_row_interactions_timer is not None:
                    if batch_index % 50 == 0:
                        loss_dict = self.loss.get_intermediate_epoch_losses()
                        loss_dicts = {
                            'train_loss': {},
                            'val_loss': {},
                            'test_loss': {}}
                        loss_dicts[f'{dataset_mode}_loss'] = loss_dict
                        self.logger.log(
                            steps=self.scheduler.num_steps, epoch=epoch,
                            **loss_dicts)

                    if self.check_row_corruption_timer():
                        break

            else:
                # Normal execution  Odhran: normal execution is here for run_epoch!!!
                self.run_batch(
                    batch_dict_, dataset_mode, eval_model,
                    epoch, print_n, batch_index)

        if self.c.image_test_patch_permutation != 0 and dataset_mode == 'train':
            return {}

        # Perform batch GD? Odhran - that's gradient descent (GD)
        batch_GD = (dataset_mode == 'train') and (
            not self.c.exp_minibatch_sgd)

        if eval_model or batch_GD:
            # We want loss_dict either for logging purposes
            # or to backpropagate if we do full batch GD
            loss_dict = self.loss.finalize_epoch_losses(eval_model)

        # (See docstring) Either perform full-batch GD (as here)
        # or mini-batch SGD (in run_batch)
        if (not eval_model) and batch_GD:
            # Backpropagate on the epoch loss
            train_loss = loss_dict['total_loss']
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.tradeoff_annealer is not None:
                self.tradeoff_annealer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        # Reset batch and epoch losses
        self.loss.reset()

        # Always return loss_dict
        # - If we are doing minibatching, return None to signify we must
        #       perform another set of mini-batch forward passes over train
        #       entries to get an eval loss.
        # - If we are doing full-batch training, we return the loss dict to
        #       immediately report loss metrics at eval time.
        if (not eval_model) and self.c.exp_minibatch_sgd:
            loss_dict = None

        return loss_dict

    def run_batch(self, batch_dict, dataset_mode, eval_model,
                  epoch, print_n, batch_index):
        # print('debug', epoch, batch_index, dataset_mode, batch_dict['data_arrs'][0].shape)
        # In stochastic label masking, we actually have a separate
        # label_mask_matrix. Else, it is just None.
        masked_tensors, label_mask_matrix, augmentation_mask_matrix = ( 
            batch_dict['masked_tensors'], 
            batch_dict['label_mask_matrix'], #odhran - mask of labels
            batch_dict['augmentation_mask_matrix']) #odhran - mask of features

        if self.c.debug_label_leakage:
            debug.leakage(
                self.c, batch_dict, masked_tensors, label_mask_matrix,
                dataset_mode)

        # Construct ground truth tensors
        ground_truth_tensors = batch_dict['data_arrs']

        if not self.c.data_set_on_cuda:
            if self.is_distributed:
                device = self.gpu
            else:
                device = self.c.exp_device

            # non_blocking flag is appropriate when we are pinning memory
            # and when we use Distributed Data Parallelism

            # If we are fitting the full dataset on GPU, the following
            # tensors are already on the remote device. Otherwise, we can
            # transfer them with the non-blocking flag, taking advantage
            # of pinned memory / asynchronous transfer.

            # Cast tensors to appropriate data type
            ground_truth_tensors = [
                torch_cast_to_dtype(obj=data, dtype_name=self.c.data_dtype) #Odhran - cast your tensors and send them to the GPU
                for data in ground_truth_tensors]
            ground_truth_tensors = [
                data.to(device=device, non_blocking=True)
                for data in ground_truth_tensors]
            masked_tensors = [
                data.to(device=device, non_blocking=True)
                for data in masked_tensors]

            # Send everything else used in loss compute to the device
            batch_dict[f'{dataset_mode}_mask_matrix'] = (
                batch_dict[f'{dataset_mode}_mask_matrix'].to(
                    device=device, non_blocking=True))

            if augmentation_mask_matrix is not None:
                augmentation_mask_matrix = augmentation_mask_matrix.to(
                    device=device, non_blocking=True)

            # Need label_mask_matrix for stochastic label masking
            if label_mask_matrix is not None:
                label_mask_matrix = label_mask_matrix.to(
                    device=device, non_blocking=True)

        forward_kwargs = dict(  #Odhran - these are the arguments we are going to pass to the forward pass
            batch_dict=batch_dict,
            ground_truth_tensors=ground_truth_tensors,
            masked_tensors=masked_tensors, dataset_mode=dataset_mode,
            eval_model=eval_model, epoch=epoch,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix)

        # This Automatic Mixed Precision autocast is a no-op
        # of c.model_amp = False
        with torch.cuda.amp.autocast(enabled=self.c.model_amp):
            if (eval_model and self.c.exp_conditional_evaluation and
                    dataset_mode in ['val', 'test']):
                self.conditional_forward_and_loss(**forward_kwargs)  #Odhran - can ignore conditional for now
            else:
                self.forward_and_loss(**forward_kwargs) #Odhran - DO NOT SKIP THIS - THIS DOES THE FORWARD TO FIND THE LOSS

        # (See docstring) Either perform mini-batch SGD (as here)
        # or full-batch GD (as further below)
        if (dataset_mode == 'train' and self.c.exp_minibatch_sgd
                and (not eval_model)):
            # Standardize and backprop on minibatch loss
            # if minibatch_sgd enabled
            loss_dict = self.loss.finalize_batch_losses()
            train_loss = loss_dict['total_loss']

            # ### Apply Automatic Mixed Precision ###
            # The scaler ops will be no-ops if we have specified
            # c.model_amp is False in the Trainer init

            # Scales loss.
            # Calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(train_loss).backward()

            # scaler.step() first unscales the gradients of the
            # optimizer's assigned params.
            # If these gradients do not contain infs or NaNs,
            # optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

            if self.tradeoff_annealer is not None:
                self.tradeoff_annealer.step()   #step the tradeoff anealer

            self.scheduler.step()
            self.optimizer.zero_grad()

            if print_n and (self.scheduler.num_steps % print_n == 0):
                self.logger.intermediate_log(  #this logs to wandb
                    loss_dict=loss_dict,
                    num_steps=self.scheduler.num_steps,
                    batch_index=batch_index, epoch=epoch)

        # Update the epoch loss info with detached minibatch losses
        self.loss.update_losses(eval_model=eval_model)

    def mc_forward_passes(  #Odhran - can be safely ignored
            self, batch_dict, dataset_mode, label_mask_matrix,
            augmentation_mask_matrix, masked_tensors):
        """
        Computes stochastic forward passes by resampling augmentation masks.
        Predicted values for label entries:
            are averaged over the T forward passes (normalized by T).
        Predicted values for augmentation entries:
            are averaged over the K <= T forward passes for which a mask
            was actually sampled at the respective entry (normalized by K).

        :param batch_dict: Used to resample masks for each forward pass
        :param dataset_mode: Restricts the sampling to a subset of data in the
            non-semisupervised setting.
        :param label_mask_matrix: label masks already sampled for the 1st pass.
        :param augmentation_mask_matrix:
            augmentation masks already sampled for the 1st pass.
        :param masked_tensors:
            model input_tensors already masked for the 1st pass.
        :return:
            output averaged across the forward passes, as detailed above.
        """
        # We will continually update these with additional forward passes,
        # as we sample entries that have yet to be masked
        # (We update label masks for compatibility with stochastic labeling)
        label_mask_entries = label_mask_matrix.to(
            device=self.c.exp_device)
        augmentation_mask_entries = augmentation_mask_matrix.to(
            device=self.c.exp_device)

        output = self.model(masked_tensors)

        def filter_output(output_matrix, label_masks, aug_masks):
            """
            We immediately filter the outputs to the augmentation and label
            masked entries of forward pass t \in T.
            """
            entries_to_preserve = (label_masks | aug_masks)

            for col_index in range(len(output)):
                col_entries_to_preserve = entries_to_preserve[:, col_index]
                output_col = output_matrix[col_index]

                # Set everything but our entries to preserve to zero
                output_col[~col_entries_to_preserve, :] = 0

            return output_matrix

        def update_outputs(running_sum, new_filtered_output):
            """
            Update outputs with new predictions from forward pass t \in T.
            """
            for col_index in range(len(new_filtered_output)):
                running_sum[col_index] += new_filtered_output[col_index]

            return running_sum

        summed_output = filter_output(
            output_matrix=output, label_masks=label_mask_matrix,
            aug_masks=augmentation_mask_matrix)

        # We immediately sum the new (and pruned) output, along with the
        #   label and augmentation mask entries,
        #   to avoid memory cost \in O(T*N*M).
        for t in range(1, self.c.model_num_eval_mc_samples):
            # Resample masks
            masked_tensors, label_mask_matrix, augmentation_mask_matrix = (
                self.mask_data(batch_dict, dataset_mode))

            # Transfer to device
            label_mask_matrix = label_mask_matrix.to(
                device=self.c.exp_device)
            augmentation_mask_matrix = augmentation_mask_matrix.to(
                device=self.c.exp_device)

            # Perform forward pass t
            pass_t_output = self.model(masked_tensors)

            # Filter output to the augmentation and label entries of this pass
            filtered_output = filter_output(
                output_matrix=pass_t_output, label_masks=label_mask_matrix,
                aug_masks=augmentation_mask_matrix)

            # Update our running sum of outputs
            summed_output = update_outputs(
                running_sum=summed_output, new_filtered_output=filtered_output)

            # Update our running sum of label and augmentation entries
            label_mask_entries += label_mask_matrix
            augmentation_mask_entries += augmentation_mask_matrix

        # Normalize the output and return
        total_mask_entries = label_mask_entries + augmentation_mask_entries

        normalized_output = []
        for col_index in range(len(summed_output)):
            summed_output_col = summed_output[col_index]
            mask_entries_col = total_mask_entries[:, col_index]
            normalized_output_col = torch.div(
                summed_output_col, mask_entries_col[:, None])
            normalized_output_col = torch.nan_to_num(
                normalized_output_col, nan=0.0, posinf=0.0)
            normalized_output.append(normalized_output_col)

        return normalized_output

    # @profile
    def forward_and_loss(
            self, batch_dict, ground_truth_tensors, masked_tensors,
            dataset_mode, eval_model, epoch, label_mask_matrix,
            augmentation_mask_matrix,):
        """Run forward pass and evaluate model loss."""

        # Check if test set predictions depend on including train set.
        # if dataset_mode == 'test':
        #     bound = data['row_boundaries']['train']
        #     for col, _ in enumerate(masked_tensors):
        #         masked_tensors[col][:bound] = 0
        #         print(masked_tensors[col])

        extra_args = {}

        if self.c.model_supcon_regularizer and dataset_mode == 'train':  #Odhran: you can completely ignore this if loop
            X_labels = ground_truth_tensors[self.supcon_target_col]
            X_labels = torch.argmax(
                torch_cast_to_dtype(obj=X_labels, dtype_name=self.c.data_dtype),
                dim=1)
            extra_args['X_labels'] = X_labels

        if eval_model:
            with torch.no_grad():
                if (self.c.model_num_eval_mc_samples > 1 and
                        dataset_mode != 'train'):
                    raise NotImplementedError(
                        'Need to update for new dataloader.')
                    assert augmentation_mask_matrix is not None, (
                        'User is attempting to use MC sampling without any '
                        'augmentation masks at eval time. Try setting '
                        '--model_augmentation_bert_mask_prob to be non-zero '
                        'for val and test.')
                    output = self.mc_forward_passes(
                        batch_dict=batch_dict, dataset_mode=dataset_mode,
                        label_mask_matrix=label_mask_matrix,
                        augmentation_mask_matrix=augmentation_mask_matrix,
                        masked_tensors=masked_tensors)
                else:
                    output = self.model(masked_tensors, **extra_args) #Odhran - this is legitimately a forward pass
        else:
            output = self.model(masked_tensors, **extra_args)

        if self.c.model_supcon_regularizer and dataset_mode == 'train': #Odhran - can ignore supcon regulariser
            output, supcon_loss = output
        else:
            supcon_loss = None

        loss_kwargs = dict(
            output=output, ground_truth_data=ground_truth_tensors,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix,
            data_dict=batch_dict, dataset_mode=dataset_mode,
            eval_model=eval_model, supcon_loss=supcon_loss)

        self.loss.compute(**loss_kwargs) #Odhran - this computes the loss

    def conditional_forward_and_loss( #Odhran - can be ignored
            self, batch_dict, ground_truth_tensors, masked_tensors,
            dataset_mode, eval_model, epoch, label_mask_matrix,
            augmentation_mask_matrix,):
        """
        Run forward pass and evaluate model loss.

        Correctly evaluate conditional likelihoods: Iteratively ...
        - ... condition on observed values
        - ... repredict with model given these observed values.
        """
        if augmentation_mask_matrix is not None:
            raise ValueError(
                'Only conditional prediction on labels is implemented!')

        # Loss compute will look at
        #   batch_dict[f'{dataset_mode}_mask_matrix']
        #   for entries on which to compute loss.
        # Just iteratively reveal values from that matrix while, at the same
        #   time, iteratively filling in the true values as input.

        # Maximum number of iterations
        missing = deepcopy(batch_dict[f'{dataset_mode}_mask_matrix'])
        missing_per_row = missing.sum(1)

        # Most missing elements in any given row
        max_missing = torch.max(missing_per_row)

        # for each row, sample an order in which to infill ground truth values
        order_per_row = [
            (torch.multinomial(1. * miss, n_miss, replacement=False)
                if n_miss > 0 else [])
            for miss, n_miss in zip(missing, missing_per_row)]

        # checked_so_far = torch.zeros_like(missing)

        for it in range(max_missing):
            if it > 0:
                # modify input to set values in selected_at_it to true values
                # and unset their mask token as well!
                for i in range(len(masked_tensors)):
                    col_select = selected_at_it[selected_at_it[:, 1] == i]
                    # column is == i for all selected
                    rows = col_select[:, 0]
                    # For each row, overwrite masked input with true values
                    # (this also unsets mask values)
                    masked_tensors[i][rows] = ground_truth_tensors[i][rows]
                    # (all values previously revealed stay revealed ofc)

            # predict with model on partially revealed input
            with torch.no_grad():
                output = self.model(masked_tensors)

            # for each row, compute loss only for current position
            # (a column index different for each row)
            # this requires generating a manipulated mask matrix
            loss_reveal = torch.zeros_like(
                batch_dict[f'{dataset_mode}_mask_matrix'])
            # (based on which loss is computed)
            # TODO: check if this works in compute

            # now reveal the first entry for each row
            # (skip rows that no longer have entries left)
            selected_at_it = torch.tensor([
                [row, per_row[it].item()]
                for row, (per_row, n_miss) in enumerate(
                    zip(order_per_row, missing_per_row))
                if it < n_miss
            ])
            loss_reveal[selected_at_it[:, 0], selected_at_it[:, 1]] = True
            # assert torch.equal(torch.nonzero(loss_reveal), selected_at_it)
            # checked_so_far += loss_reveal

            batch_dict[f'{dataset_mode}_mask_matrix'] = loss_reveal

            # and now evaluate the loss!
            loss_kwargs = dict(
                output=output, ground_truth_data=ground_truth_tensors,
                label_mask_matrix=label_mask_matrix,
                augmentation_mask_matrix=augmentation_mask_matrix,
                data_dict=batch_dict, dataset_mode=dataset_mode,
                eval_model=eval_model)

            self.loss.compute(**loss_kwargs)

        # check at the end that all values have been input
        # assert torch.equal(checked_so_far, missing)

    def eval_check(self, epoch):
        """Check if it's time to evaluate val and test errors."""
        # check number of epochs and steps 
        # tells us when to do eval vs test. 
        if self.c.exp_eval_every_epoch_or_steps == 'epochs':
            return epoch % self.c.exp_eval_every_n == 0
        elif self.c.exp_eval_every_epoch_or_steps == 'steps':
            # Cannot guarantee that we hit modulus directly.
            if (self.scheduler.num_steps - self.last_eval >=
                    self.c.exp_eval_every_n):
                self.last_eval = self.scheduler.num_steps
                return True
            else:
                return False
        else:
            raise ValueError

    def set_row_corruption_timer(self):
        assert self.c.debug_eval_row_interactions_timer is not None
        self.row_corruption_timer = time.time()
        self.n_row_corr_batches = 0

    def check_row_corruption_timer(self):
        break_loop = False
        self.n_row_corr_batches += 1
        n_examples = self.n_row_corr_batches * self.c.exp_batch_size
        print(f'Row Corruption: completed {self.n_row_corr_batches} batches, '
              f'{n_examples} examples.')

        if (time.time() - self.row_corruption_timer >
                (self.c.debug_eval_row_interactions_timer * 60 * 60)):
            print(f'Row Corruption: have reached time limit: '
                  f'{self.c.debug_eval_row_interactions_timer} hours.')
            print('Breaking loop.')
            break_loop = True

        return break_loop
