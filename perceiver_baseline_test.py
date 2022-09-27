from functools import partial
from tokenize import group
import wandb
from z_cifar_npt_files.metrics import print_metrics_binary, print_metrics_multilabel
from z_cifar_npt_files.perceiver_pytorch import Perceiver, IterativePerceiver
from z_cifar_npt_files.rnns import RNN, LSTM, GRU
from z_cifar_npt_files.data_loaders import *
# from z_cifar_npt_files.perceiver_io import PerceiverIO
from z_cifar_npt_files.embedding_generators import *
from z_cifar_npt_files.losses import *
import typer
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy
from pytorch_lightning.loggers import WandbLogger
import torch_optimizer as optim
from dotmap import DotMap
from time import strftime

class LigthningPerceiver(pl.LightningModule):

    def __init__(
            self, c, logger, time_code, device):
        super().__init__()
        self.c = c
        self.table_logger = logger
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        if c.num_classes == 2:
            self.val_auroc = AUROC(num_classes=2)

        self.arrange_data = partial(
            arrange_data, structure = c['structure'])

        expl_data = self.arrange_data(c['expl_batch'])
        self.input_axes = len(expl_data.shape) - 2

        if self.c.no_embedding:
            self.c.embedding_dim = expl_data.shape[-1]
            self.embedding = nn.Identity()
        else:
            embedding_modules = []
            if c.data_type == 'catagorical' and self.c.per_item_embed:
                embedding_modules.append(CategoricalEmbeddingGenerator(
                    self.c.hidden_dim, time_code))
            elif c.data_type == 'catagorical' and not self.c.per_item_embed:    
                embedding_modules.append(torch.nn.Embedding(
                    self.c.feature_dim, self.c.embedding_dim))
            elif c.data_type == 'continuous' and self.c.per_item_embed:
                raise NotImplementedError()
            elif c.data_type == 'continuous' and not self.c.per_item_embed:
                embedding_modules.append(nn.Linear(
                    self.c.feature_dim, self.c.embedding_dim))
            else:
                raise NotImplementedError()

            if time_code and self.c.feature_embed:
                embedding_modules.append(Feature_embedding(
                    self.c.hidden_dim, time_code, device))
            if time_code and self.c.time_embed:
                embedding_modules.append(Time_embedding(
                    self.c.hidden_dim, time_code, device))            
            self.embedding = nn.Sequential(*embedding_modules)

        if self.c.model_type == 'Perceiver':
            input_model = Perceiver
        elif self.c.model_type == 'Iterative-Perceiver':
            input_model = IterativePerceiver
        elif self.c.model_type == 'LSTM':
            input_model = LSTM
        elif self.c.model_type == 'RNN':
            input_model = RNN
        elif self.c.model_type == 'GRU':
            input_model = GRU
        else:
            NotImplementedError()

        if c.loss_type == 'MSE':
            assert not c.multilabel
            self.logits = nn.Softmax()
            self.loss = partial(
                mean_squared_error,  num_classes=c.num_classes)
        elif c.loss_type == 'multi-label-CE':
            assert c.num_classes > 2 and c.multilabel
            self.logits = nn.Sigmoid()
            self.loss = multi_class_multi_label_CE_loss
        elif c.loss_type == 'NLL':
            assert not c.multilabel
            self.logits = nn.LogSoftmax()
            self.loss = negative_log_likelihood
        elif c.loss_type == 'Weighted-CE':
            assert not c.multilabel
            self.logits = nn.Sigmoid()
            self.loss = single_label_weighted_ce_loss
        else:
            NotImplementedError()

        self.model = input_model(
            input_channels=self.c.embedding_dim,          # number of channels for each token of the input
            input_axis=self.input_axes,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands=6,          # number of freq bands, with original value (2 * K + 1)
            max_freq=10.,              # maximum frequency, hyperparameter depending on how fine the data is
            time_code=time_code,
            depth=self.c.layers,                   # depth of net. The shape of the final attention mechanism will be:
                                        #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=self.c.num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=self.c.hidden_dim,            # latent dimension
            cross_heads=1,             # number of heads for cross attention. paper said 1
            latent_heads=self.c.num_heads,            # number of heads for latent self attention, 8
            cross_dim_head=64,         # number of dimensions per cross attention head
            latent_dim_head=self.c.head_dim,        # number of dimensions per latent self attention head
            num_classes=self.c.num_classes,          # output number of classes
            attn_dropout=self.c.dropout,
            ff_dropout=self.c.dropout,
            weight_tie_layers=False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=bool(self.c.use_fourrier),  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=2      # number of self attention blocks per cross attention
        )

    def configure_optimizers(self):
        if self.c.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.c.lr)
        elif self.c.optimizer == 'Lamb':
            optimizer = optim.Lamb(
                self.parameters(), lr=self.c.lr,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        elif self.c.optimizer == 'Lookahead-Lamb':
            Lamb = optim.Lamb(
                self.parameters(), lr=self.c.lr,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            optimizer = optim.Lookahead(Lamb, k=0.5, alpha=0.5)

        return optimizer

    def training_step(self, batch, batch_idx):
        _, y, _, logits, loss = self._common_step(batch, batch_idx, "train")
        if self.c.multilabel:
            self.train_accuracy(logits, y.long())
        else:
            if len(y.shape) == 2:
                y = torch.squeeze(y.long())    
            self.train_accuracy(torch.argmax(logits, dim=-1), y)
        self.log('train_accuracy', self.train_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y, _, logits, loss = self._common_step(batch, batch_idx, "val")
        if self.c.multilabel:
            self.val_accuracy(logits, y.long())
            self.log(
                'multilabel_val_performance',
                print_metrics_multilabel(y.cpu(), logits.cpu(), verbose=0))            
            i=0
            for l_i, y_i in zip(logits.T.cpu(), y.T.cpu()):
                self.log(
                    'c{}_binary_val_performance'.format(i),
                    print_metrics_binary(y_i, l_i, verbose=0))
                i += 1
        else:
            if len(y.shape) == 2:
                y = torch.squeeze(y.long())    
            self.val_accuracy(torch.argmax(logits, dim=-1), y)
        self.log('val_accuracy', self.val_accuracy)
        if self.c.num_classes == 2:
            self.val_auroc(logits, y)
            self.log('val_auroc', self.val_auroc)
        return loss

    def _log_x_and_y(self, z, y, stage: str):
        columns = ['prediction', 'target']
        data = list(map(list, zip(torch.argmax(z, dim=-1), y)))
        self.table_logger.log_table(
            key=f"{stage}_samples", columns=columns, data=data)


    def _common_step(self, batch, batch_idx, stage: str):
        x, y = batch
        x = self.arrange_data(x)
        # x = torch.unsqueeze(x, -1) if self.c.no_embedding else x

        if self.c.data_type == 'categorical':
            x, y = x.long(), y.long()

        x = self.embedding(x)
        z = self.model(x)
        logits = self.logits(z)
        loss = self.loss(pred=logits, target=y)

        self.log(f"{stage}_loss", loss, on_step=True)
        return x, y, z, logits, loss


def setup(
        # non-model
        project: str = None,
        dataset: str = 'mimic-pheno', # mnist, mimic-pheno uci-breast
        batch_size: int = 128,
        max_epochs: int = 100,
        dev_run: int = 0,
        use_16_bit: int = 1,
        # model
        model_type: str = 'LSTM', # Iterative-Perceiver, RNN, GRU, LSTM, Pereceiver
        optimizer: str = 'Lamb',
        loss_type: str = 'multi-label-CE', # multi-label-CE, NLL, Weighted-CE, MSE
        dropout: float = 0.0,
        lr: float = 1e-3,
        layers: int = 2,
        num_heads:  int = 8,
        head_dim: int = 64,
        embedding_dim:  int = 128,
        num_latents: int = 256,
        per_item_embed: int = 0,
        feature_embed: int = 0,
        time_embed: int = 0,
        undersample: int = 0,
        use_fourrier: int = 1,
        no_embedding: int = 1,
        cv_splits: int = 1,
        ):

    if project == None:
        project = '-'.join(
            [model_type,dataset,optimizer,loss_type])

    if cv_splits == 1:
        wandb_logger = WandbLogger(project=project)
    else:
        group = 'sweep-at-'+ strftime("%d-%b-%H-%M-%S")
        wandb_logger = WandbLogger(
            project=project, 
            group=group)

    c = {
        'model_type': model_type, 'optimizer': optimizer,
        'loss_type': loss_type, 'dropout': dropout, 'lr': lr,
        'layers': layers, 'num_heads': num_heads, 'head_dim': head_dim,
        'hidden_dim': num_heads*head_dim,
        'embedding_dim': embedding_dim, 'num_latents': num_latents,
        'per_item_embed': per_item_embed, 'time_embed': time_embed,
        'feature_embed': feature_embed, 'undersample': undersample,
        'use_fourrier': use_fourrier, 'no_embedding': no_embedding,
        'multilabel': False,
    }

    if dataset == 'mimic':
        train_loader, val_loader, num_features, num_targets, time_code, structure = load_mimic(
            batch_size=batch_size, undersample=undersample)
        c['data_type'] = 'categorical'

    elif dataset == 'mnist':
        train_loader, val_loader, num_features, num_targets, time_code, structure = load_mnist(
            batch_size=batch_size)       
        c['data_type'] = 'continuous'

    elif dataset == 'mimic-pheno':
        train_loader, val_loader, num_features, num_targets, time_code, structure = load_mimic_pheno(
           batch_size=batch_size, shuffle=False)
        c['data_type'] = 'continuous'
        c['multilabel'] = True

    elif 'uci' in dataset:
        train_loader, val_loader, num_features, num_targets, time_code, structure = load_uci(
            batch_size=batch_size,
            dataset=dataset)
        c['data_type'] = 'continuous'
    else:
        raise NotImplementedError()

    c['feature_dim'] = num_features
    c['num_classes'] = num_targets
    c['structure'] = structure
    c['expl_batch'] = next(iter(train_loader))[0]


    print(c)


    model = LigthningPerceiver(
            DotMap(c),
            logger=wandb_logger,
            time_code=time_code,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
    if dev_run:
        trainer = pl.Trainer(fast_dev_run=True)
    else:
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            precision=16 if torch.cuda.is_available() and use_16_bit else 32,
            logger=wandb_logger,
            max_epochs=max_epochs)

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    typer.run(setup)