# ---------------------------------- #
# refine the Amiya Class by Deepseek & Gork3
# using GATv2Conv replace SuperConv to reduces complexity
# ---------------------------------- #

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import random
import os
import pickle as pkl
import re
import datetime
from tqdm import tqdm
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, TransformerEncoder, TransformerEncoderLayer, CosineEmbeddingLoss, MSELoss, Parameter
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torcheval.metrics import MulticlassAUROC
from sklearn.model_selection import train_test_split

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINConv, GATv2Conv
from torch_geometric.nn.norm import BatchNorm, GraphNorm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, mask_feature, mask_to_index, index_to_mask, to_dense_batch, select
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation, SetTransformerAggregation

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import ConcordanceCorrCoef
from lightning.pytorch.profilers import PyTorchProfiler


import importlib
import sys
sys.path.append('/mnt/Venus/home//liuzhaoyang/project/MetaSpace/')
import processing as MSpp
import model as MS

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)



class MetAggregator(nn.Module):
    def __init__(self, channels: int, nfeature_embed: int, n_spot_max: int):
        super().__init__()
        self.n_spot_max = n_spot_max
        # self.proj_in = nn.Linear(n_feature, hidden_dim)
        # Initialize GraphMultisetTransformer
        self.set_aggr = SetTransformerAggregation(
            channels=channels,
            num_seed_points=nfeature_embed,
            heads=1,
            dropout=0.2,
            num_encoder_blocks=1,
            num_decoder_blocks=1
        )
        # self.proj_out = nn.Linear(hidden_dim, nfeature_embed)

    def forward(self, X_met: torch.Tensor, spot_counts: torch.Tensor) -> torch.Tensor:
        """
        Aggregate X_met and reshape to [batch_size * n_spot, nfeat_embed], removing zero-paddings.

        Args:
            X_met (torch.Tensor): Input tensor of shape [batch_size, n_feature, n_spot_max]
            spot_counts (torch.Tensor): Number of actual spots per batch, shape [batch_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size * n_spot, nfeat_embed]
        """
        batch_size, n_feature, n_spot_max = X_met.shape
        device = X_met.device

        batch = torch.repeat_interleave(torch.arange(batch_size, device=device), n_feature)
        X_met_flat = X_met.reshape(batch_size*n_feature, n_spot_max)
        
        with torch.cuda.amp.autocast():
            X_met_aggr = self.set_aggr(X_met_flat, index=batch)
            # print('> MetAggregator > X_met_aggr: {}'.format(X_met_aggr.size()))

            X_met_aggr = X_met_aggr.view(batch_size, n_spot_max, -1)
            # print('> MetAggregator > X_met_aggr out: {}'.format(X_met_aggr.size()))
            
            # Remove zero-paddings using spot_counts
            X_out_list = []
            for i in range(batch_size):
                # Select only the valid spots for this batch
                X_out_list.append(X_met_aggr[i, :spot_counts[i], :])
    
            # Concatenate to get [total_spots, nfeat_embed]
            X_out_final = torch.cat(X_out_list, dim=0)
            # print('> MetAggregator > X_out_final: {}'.format(X_out_final.size()))

        return X_out_final



#########
# model #
#########

class Amiya(nn.Module):
    def __init__(
        self,
        n_gene: int,
        n_rxn: int,
        nfeature_embed: int = 512,
        nlayer_ST_graph: int = 2,
        nlayer_met_graph: int = 2,
        nfeature_met: int = 14,
        nhead_SGAT: int = 4,
        masking_rate: float = 0.5,
        replace_rate: float = 0.1,
        GIN_eps: float = 0,
        MLP_dr: float = 0.1,
        MLP_act: str = 'gelu',
        MLP_norm: str = 'GraphNorm',
        train_GIN_eps: bool = False
    ):
        super().__init__()
        self.n_gene = n_gene
        self.n_rxn = n_rxn
        self.nfeature_embed = nfeature_embed
        self.nfeature_met = nfeature_met
        self.masking_rate = masking_rate
        self.replace_rate = replace_rate
        self.nhead_SGAT = nhead_SGAT
        self.nlayer_ST_graph = nlayer_ST_graph
        self.nlayer_met_graph = nlayer_met_graph
        
        # Initialize layers
        self._init_masking_parameters()
        self._init_normalization()
        self._init_projection_layers()
        self._init_graph_layers(nlayer_ST_graph, nlayer_met_graph, GIN_eps, train_GIN_eps, 
                              MLP_dr, MLP_act, MLP_norm)
        self._init_aggregation_layers(nlayer_met_graph)
        self._init_mapping_layers(MLP_dr, MLP_act, MLP_norm, nlayer_met_graph)

    def _init_masking_parameters(self):
        self.spot_mask_token = nn.Parameter(torch.zeros(1, self.n_gene)) 
        # change: init size using the max avi n_spots
        # but the batch info cannot access here, 
        self.feature_mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.nfeature_embed)) for _ in range(self.nlayer_ST_graph + self.nlayer_met_graph)
        ])

    def _init_normalization(self):
        self.input_gn = GraphNorm(self.n_gene)
        self.embed_gn = GraphNorm(self.nfeature_embed)

    def _init_projection_layers(self):
        self.identity_proj = nn.Linear(self.n_gene, self.nfeature_embed)
        self.identity_met_proj = nn.Linear(self.n_gene, self.n_rxn)

    def _init_graph_layers(self, n_ST, n_met, eps, train_eps, dr, act, norm):
        def _mlp(in_dim, out_dim):
            return MLP([in_dim, 256, 128, 256, out_dim], dr, act, norm)

        self.ST_layers = nn.ModuleList([
            GINConv(_mlp(self.n_rxn if i == 0 else self.nfeature_embed, self.nfeature_embed), 
                   eps=eps, train_eps=train_eps)
            for i in range(n_ST)
        ])

        self.met_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=self.nfeature_met,
                out_channels=self.nfeature_met,
                heads=self.nhead_SGAT,
                concat=False,
                dropout=0.5,
                add_self_loops=False,
                negative_slope=0.2
            ) for _ in range(n_met)
        ])

    def _init_aggregation_layers(self, n_met):
        # self.met_aggrs = nn.ModuleList([
        #     GraphMultisetTransformer(
        #         channels=self.nfeature_met,
        #         k=self.nfeature_embed,
        #         num_encoder_blocks=2,
        #         heads=1,
        #         dropout=0.2
        #     ) for _ in range(n_met)
        # ])
        self.met_aggrs = nn.ModuleList([
            MetAggregator(
                channels=self.nfeature_met,
                nfeature_embed=self.nfeature_embed,
                n_spot_max=self.nfeature_met
            ) for _ in range(n_met)
        ])

    def _init_mapping_layers(self, dr, act, norm, n_met):
        self.gene2rxn = MLP([self.n_gene, 2048, 1024, 256, self.n_rxn], dr, act, norm)
        self.embed_mapper = MLP(
            [self.nfeature_embed * (self.nlayer_ST_graph + n_met), 512, 128, 512, self.n_rxn],
            0.3, 'gelu', 'GraphNorm'
        )
        self.rxn2gene = MLP([self.n_rxn, 256, 1024, 2048, self.n_gene], dr, act, norm)

    def select_mask_index(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        counts = torch.bincount(batch)
        offsets = torch.cat([torch.zeros(1, device=self.device), counts.cumsum(dim=0)[:-1]])
        mask_counts = (counts.float() * self.masking_rate).long()
        mask_indices = [offset + torch.randperm(c, device=self.device)[:k] 
                        for offset, c, k in zip(offsets, counts, mask_counts)]
        mask_index = torch.cat(mask_indices).long()
        # print('> select_mask_index > mask_index: {}'.format(mask_index.size()))
        keep_mask = torch.ones(len(batch), dtype=torch.bool, device=self.device)
        keep_mask[mask_index] = False
        # print('> select_mask_index > keep_mask: {}'.format(keep_mask.size()))
        return mask_index, torch.where(keep_mask)[0]

    def spot_masking(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, tuple]:
        mask_idx, keep_idx = self.select_mask_index(batch)
        # print('> spot_masking > x: {}'.format(x.size()))
        # print('> spot_masking > batch: {}'.format(batch.size()))
        # print('> spot_masking > mask_idx: {}'.format(mask_idx.size()))
        # print('> spot_masking > keep_idx: {}'.format(keep_idx.size()))

        mask_idx_bool = index_to_mask(mask_idx, size=x.size(0))
        # print('> spot_masking > spot_mask_token: {}'.format(self.spot_mask_token.size()))
        # print('> spot_masking > mask_idx_bool: {}'.format(mask_idx_bool.size()))
        # print('> spot_masking > where a: {}'.format(mask_idx_bool[:, None].expand(-1, x.size(1)).size()))
        # print('> spot_masking > where b: {}'.format(self.spot_mask_token.expand(x.size(0), x.size(1)).size()))
        x_masked = torch.where(mask_idx_bool[:, None].expand(-1, x.size(1)).to(x.device), 
                              self.spot_mask_token.expand(x.size(0), x.size(1)), 
                              x)
        return x_masked, (mask_idx, keep_idx)

    def feature_masking(self, x: Tensor) -> Tuple[Tensor, tuple]:
        mask = torch.rand(x.shape[1], device=self.device) < self.masking_rate
        # print('> feature_masking > mask: {}'.format(mask.size()))
        x_masked = x.clone()
        x_masked[:, mask] = 0
        if self.replace_rate > 0:
            replace_mask = mask & (torch.rand_like(mask.float()) < self.replace_rate)
            perm = torch.randperm(x.shape[1], device=self.device)[:replace_mask.sum()]
            x_masked[:, replace_mask] = x[:, perm]
        return x_masked, (torch.where(mask)[0], torch.where(~mask)[0])

    def reshape_to_met(self, X: torch.Tensor, batch: torch.Tensor, n_spot_max: int, n_feature: int) -> torch.Tensor:
        """
        Reshape X from [total_spots, n_feature] to [batch_size, n_spot_max, n_feature],
        padding with zeros for batches where n_spot < n_spot_max.
    
        Args:
            X (torch.Tensor): Input tensor of shape [total_spots, n_feature].
            batch (torch.Tensor): Batch assignment tensor of shape [total_spots], with values in [0, batch_size-1].
            n_spot_max (int): Maximum number of spots across all batches.
            n_feature (int): Number of features per spot.
    
        Returns:
            torch.Tensor: Reshaped and padded tensor of shape [batch_size, n_spot_max, n_feature].
        """
        # Determine batch_size from the batch tensor (assuming batch indices start at 0)
        batch_size = batch.max().item() + 1
        device = X.device
    
        # Initialize output tensor with zeros for padding
        X_met = torch.zeros(batch_size, n_spot_max, n_feature, device=device)
    
        # Count the number of spots per batch
        spot_counts = torch.bincount(batch, minlength=batch_size)
    
        # Compute start indices for each batch in the flattened X
        cumsum = torch.cumsum(spot_counts, dim=0)
        start_idx = torch.cat([torch.tensor([0], device=device), cumsum[:-1]])
    
        # Assign spots to X_met for each batch
        for i in range(batch_size):
            # Extract spots for batch i
            try:
                graph_spots = X[start_idx[i]:start_idx[i] + spot_counts[i]]
                # Assign to X_met; remaining positions (up to n_spot_max) stay zero
                X_met[i, :spot_counts[i]] = graph_spots
            except:
                print('> reshape_to_met > graph_spots: {}'.format(graph_spots.size()))
                print('> reshape_to_met > X_met: {}'.format(X_met.size()))
                print('> reshape_to_met > batch_size: {}'.format(batch_size))
                print('> reshape_to_met > i: {}'.format(i))
                print('> reshape_to_met > start_idx: {}'.format(start_idx[i]))
                print('> reshape_to_met > spot_counts: {}'.format(spot_counts[i]))
                print('> reshape_to_met > X_met[i, :spot_counts[i]]: {}'.format(X_met[i, :spot_counts[i]]))
                print('> reshape_to_met > graph_spots: {}'.format(graph_spots))
                print('> reshape_to_met > spot_counts: {}'.format(spot_counts))
                raise
    
        return X_met.permute(0, 2, 1), spot_counts
    

    def forward(self, x, edge_index, rxn_exp, met_edge_index, batch, batch_size, met_batch):
        # print('> forward > batch: {}'.format(batch.size()))
        # print('> forward > met_batch: {}'.format(met_batch.size()))
        # print('> forward > batch_size: {}'.format(batch_size))
        # print('> forward > x before_gn: {}'.format(x.size()))
        x = self.input_gn(x, batch=batch)
        # print('> forward > x after_gn: {}'.format(x.size()))
        x, (spot_mask, spot_keep) = self.spot_masking(x, batch)
        # print('> forward > spot_mask: {}'.format(spot_mask.size()))
        # print('> forward > spot_keep: {}'.format(spot_keep.size()))
        x, (feat_mask, feat_keep) = self.feature_masking(x)
        # print('> forward > feat_mask: {}'.format(feat_mask.size()))
        # print('> forward > feat_keep: {}'.format(feat_keep.size()))
        
        x_rxn = F.gelu(self.gene2rxn(x))
        # print('> forward > x_rxn: {}'.format(x_rxn.size()))

        with torch.cuda.amp.autocast():
            st_embeds = []
            current = x_rxn
            for i, layer in enumerate(self.ST_layers):
                current = layer(current + (self.identity_proj(x) if i > 0 else 0), edge_index)
                current = F.gelu(self.embed_gn(current))
                current[spot_mask] = self.feature_mask_tokens[i]
                # print('> forward > current: {}'.format(current.size()))
                st_embeds.append(current)
            # print('> forward > st_embeds: {}'.format(len(st_embeds)))
            
            met_embeds = []
            current_met, spot_counts = self.reshape_to_met(x_rxn, batch, self.nfeature_met, self.n_rxn)
            met_size = current_met.size()
            current_met_flat = current_met.reshape(-1,met_size[-1])
            # current_met = x_rxn.view(batch_size, -1, self.n_rxn)[:, :self.nfeature_met].permute(0, 2, 1)
            # print('> forward > current_met: {}'.format(current_met.size()))
            # print('> forward > current_met_flat: {}'.format(current_met_flat.size()))
            # print('> forward > met_edge_index: {}'.format(met_edge_index.size()))
            # print('> forward > feature_mask_tokens: {}'.format(self.feature_mask_tokens[len(self.ST_layers)].size()))
            for i, layer in enumerate(self.met_layers):
                current_met_flat = layer(current_met_flat, met_edge_index)
                current_met_flat = F.gelu(current_met_flat)
                # print('> forward > current_met_flat loop: {}'.format(current_met_flat.size()))
                met_embeds.append(current_met_flat.reshape(met_size))
            # print('> forward > met_embeds: {}'.format(len(met_embeds)))
                  
            st_concat = torch.cat(st_embeds, dim=1)
            met_concat = torch.cat([aggr(e, spot_counts) for aggr, e in zip(self.met_aggrs, met_embeds)], dim=1)
            # print('> forward > st_concat: {}'.format(st_concat.size()))
            # print('> forward > met_concat: {}'.format(met_concat.size()))
            
            combined = torch.cat([st_concat, met_concat], dim=1)
            # print('> forward > combined: {}'.format(combined.size()))
            out_rxn = F.gelu(self.embed_mapper(combined))
            # print('> forward > out_rxn: {}'.format(out_rxn.size()))
            out_gene = self.rxn2gene(out_rxn).relu()
            # print('> forward > out_gene: {}'.format(out_gene.size()))
        
        return out_gene, (spot_mask, spot_keep, feat_mask, feat_keep)

    @property
    def device(self):
        return next(self.parameters()).device

    def sce_loss(self, x: Tensor, y: Tensor, alpha: float = 3) -> Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        return (1 - (x_norm * y_norm).sum(-1)).pow(alpha).mean()




# 辅助模块定义
class MLP(nn.Module):
    def __init__(self, dims, dropout=0.1, act='gelu', norm='GraphNorm'):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.extend([
                Linear(dims[i], dims[i+1]),
                self._get_norm(norm, dims[i+1]),
                self._get_activation(act),
                nn.Dropout(dropout)
            ])
        self.net = nn.Sequential(*layers[:-1])  # 移除最后的dropout

    def _get_activation(self, act):
        return getattr(nn, act)() if hasattr(nn, act) else nn.ReLU()

    def _get_norm(self, norm, dim):
        return GraphNorm(dim) if norm == 'GraphNorm' else nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.net(x)





class MyDataLoader(NeighborLoader):
    """优化后的数据加载器"""
    def __init__(self,
                 graph_data: Data,
                 num_neighbors: list,
                 batch_size: int,
                 disjoint: bool = True,
                 num_workers: int = 10,
                 pin_memory: bool = None,
                 drop_last: bool = False,
                 **kwargs):
        # 自动判断是否启用pin_memory
        pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()
        self.graph_data = graph_data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.disjoint = disjoint
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        super().__init__(
            data=self.graph_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            disjoint=self.disjoint,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            **kwargs
        )


class LitAmiya(L.LightningModule):
    def __init__(self, met_edge_index: torch.Tensor, lr: float, wd: float, record_flag: bool,
                 n_gene: int, n_rxn: int, nfeature_embed: int = 256, nlayer_ST_graph: int = 6,
                 nlayer_met_graph: int = 6, nhead_SGAT: int = 4, nhead_mapping: int = 4,
                 batch_size: int = 64, masking_rate: float = 0.5, replace_rate: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer('met_edge_index', met_edge_index.t().contiguous())
        self.amiya = Amiya(
            n_gene, n_rxn, nfeature_embed, nlayer_ST_graph, nlayer_met_graph,
            nhead_SGAT=nhead_SGAT, masking_rate=masking_rate, replace_rate=replace_rate
        )

    def forward(self, graph: Data) -> tuple:
        # print('> LitAmiya_forward > graph.batch: {}'.format(graph.batch.size()))
        sorted_idx = torch.argsort(graph.batch)
        graph = self._reorder_graph(graph, sorted_idx)
        met_batch = torch.repeat_interleave(
            torch.arange(self.hparams.batch_size, device=self.device), self.hparams.n_rxn
        )
        # print('> LitAmiya_forward > met_batch: {}'.format(met_batch.size()))
        met_edge_index = self._prepare_met_edges()
        # print('> LitAmiya_forward > met_edge_index: {}'.format(met_edge_index.size()))
        return self.amiya(
            x=graph.x.to(self.device), edge_index=graph.edge_index.to(self.device),
            rxn_exp=graph.rxn_exp.to(self.device), 
            met_edge_index=met_edge_index, batch=graph.batch.to(self.device),
            batch_size=self.hparams.batch_size, met_batch=met_batch
        )

    def _reorder_graph(self, graph: Data, sorted_idx: torch.Tensor) -> Data:
        # print('> _reorder_graph > graph.batch: {}'.format(graph.batch.size()))
        # print('> _reorder_graph > graph.x: {}'.format(graph.x.size()))
        # print('> _reorder_graph > sorted_idx: {}'.format(sorted_idx.size()))
        idx_mapping = torch.zeros_like(sorted_idx, device=self.device)
        # print('> _reorder_graph > idx_mapping: {}'.format(idx_mapping.size()))
        idx_mapping[sorted_idx] = torch.arange(len(sorted_idx), device=self.device)
        graph.x = graph.x[sorted_idx]
        graph.pos = graph.pos[sorted_idx]
        graph.slide = graph.slide[sorted_idx]
        graph.edge_index = idx_mapping[graph.edge_index]
        graph.batch = graph.batch[sorted_idx]

        if hasattr(graph, 'rxn_exp'):
            graph.rxn_exp = graph.rxn_exp[sorted_idx]
        return graph

    def _prepare_met_edges(self) -> torch.Tensor:
        offsets = torch.arange(
            0, self.hparams.batch_size * self.hparams.n_rxn, self.hparams.n_rxn, device=self.device
        ).repeat_interleave(self.met_edge_index.size(1))
        return (self.met_edge_index.repeat(1, self.hparams.batch_size) + offsets).contiguous()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 4}}

    def _shared_step(self, graph: Data, prefix: str) -> torch.Tensor:
        out_embed, mask_info = self(graph)
        raw_embed = graph.x
        loss = self._compute_loss(out_embed, raw_embed, mask_info)
        
        self._update_metrics(out_embed, raw_embed, prefix)
        if self.hparams.record_flag:
            self.log_dict({
                f'Loss/{prefix}': loss,
                'lr': self.trainer.optimizers[0].param_groups[0]['lr']
            }, sync_dist=True, on_step=True)
        return loss

    def _compute_loss(self, pred: torch.Tensor, target: Tensor, masks: tuple) -> torch.Tensor:
        spot_mask, _, feat_mask, _ = masks
        loss_spot = self.amiya.sce_loss(pred[spot_mask], target[spot_mask])
        loss_feat = self.amiya.sce_loss(pred.T[feat_mask], target.T[feat_mask])
        # print('> _compute_loss > loss_sgat: {}'.format(self.amiya.met_layers[0].get_attention_loss().size()))
        # loss_sgat = sum([layer.get_attention_loss().to(dtype=torch.bfloat16) for layer in self.amiya.met_layers]).mean()
        return loss_spot + loss_feat

    def _update_metrics(self, pred: torch.Tensor, target: torch.Tensor, prefix: str):
        # print('> _update_metrics > pred: {}'.format(pred.size()))
        # print('> _update_metrics > target: {}'.format(target.size()))
        metrics_spot = ConcordanceCorrCoef(num_outputs=pred.size(0)).to(pred.device)
        metrics_met = ConcordanceCorrCoef(num_outputs=pred.size(1)).to(pred.device)

        # metrics_spot.update(pred.T, target.T)
        # metrics_met.update(pred, target)
        
        self.log_dict({
            f'Corr_spot/{prefix}': torch.nanmedian(metrics_spot(pred.T, target.T)),
            f'Corr_feature/{prefix}': torch.nanmedian(metrics_met(pred, target))
        }, prog_bar=True, sync_dist=True, on_step=True)

        metrics_spot.reset()
        metrics_met.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    # def on_train_epoch_end(self):
    #     self.metrics_spot.reset()
    #     self.metrics_met.reset()

    # def on_validation_epoch_end(self):
    #     self.metrics_spot.reset()
    #     self.metrics_met.reset()


class TrainingConfig:

    n_nei = 6
    nlayer_nei = 2
    batch_size = 64
    accumu_step = 5
    num_workers = 20
    
    nfeature_embed = 256
    nlayer_ST_graph = 6
    nlayer_met_graph = 6
    nhead_SGAT = 4
    masking_rate = 0.5
    replace_rate = 0.1

    max_epochs = 100
    lr = 0.001
    weight_decay = 1e-5 # weight decay
    
    accelerator='gpu'
    devices = [0,1,2,3,4,7]
    precision = "16-mixed"

    root_dir = './v0171_pre_gork/'
    
    test_size = 0.2 
    val_size = 0.1
    drop_last = True


def data_preparing(graph_list: str, met_edge: str, S_file: str, avi_meta: str, avi_genes: str):
    
    graph_list = torch.load(graph_list)
    
    # ------------------- #
    # S & metabolic graph #
    # ------------------- #
    with open(met_edge, 'rb') as f:
        met_edge = torch.tensor(pkl.load(f))
    
    # loading metabolic model info.
    S = pd.read_csv(S_file, sep='\t', index_col=0)
    S_tensor = torch.from_numpy(S.to_numpy()).float()    
    
    # ------------------------- #
    # S metabolite HMDB mapping #
    # ------------------------- #
    with open(avi_meta, 'r') as f:
        avi_meta = f.read().strip().split('\n')

    with open(avi_genes, 'r') as f:
        avi_genes = f.read().strip().split('\n')

    return graph_list, met_edge, S, S_tensor, avi_meta, avi_genes


def setup_training(cfg: TrainingConfig, graph_list: list, met_edge: torch.Tensor, n_gene: int, n_rxn: int):

    train_val_graphs, test_graphs = train_test_split(
        graph_list, 
        test_size=cfg.test_size,
        random_state=42
    )
    train_graphs, val_graphs = train_test_split(
        train_val_graphs,
        test_size=cfg.val_size/(1-cfg.test_size),
        random_state=42
    )

    train_data = merge_graphs(train_graphs)
    val_data = merge_graphs(val_graphs)
    test_data = merge_graphs(test_graphs)

    # 初始化模型
    model = LitAmiya(
        met_edge_index=met_edge,
        lr=cfg.lr,
        wd=cfg.weight_decay,
        record_flag=True,
        n_gene=n_gene,
        n_rxn=n_rxn,
        nfeature_embed=cfg.nfeature_embed,
        nlayer_ST_graph=cfg.nlayer_ST_graph,
        nlayer_met_graph=cfg.nlayer_met_graph,
        nhead_SGAT=cfg.nhead_SGAT,
        batch_size=cfg.batch_size,
        # accumu_step=cfg.accumu_step,
        masking_rate=cfg.masking_rate,
        replace_rate=cfg.replace_rate
    )
        
    return model, train_data, val_data, test_data


def get_callbacks():
    return [
        # 模型检查点
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="Loss/val",
            mode="min",
            save_top_k=2,
            filename="best-{epoch}-{Loss/val:.2f}"
        ),
        
        # 早停机制
        L.pytorch.callbacks.EarlyStopping(
            monitor="Loss/val",
            patience=15,
            mode="min"
        ),
        
        # 学习率监控
        L.pytorch.callbacks.LearningRateMonitor()
    ]


def merge_graphs(graph_list):
    """
    将多个图合并为单个大图
    :param graph_list: 图数据列表[Data]
    :return: 合并后的Data对象
    """
    offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes for g in graph_list[:-1]]),
        dim=0
    )
    
    # 合并节点特征
    x = torch.cat([g.x for g in graph_list], dim=0)
    
    # 合并边索引
    edge_indices = []
    for i, g in enumerate(graph_list):
        edge_indices.append(g.edge_index + offsets[i])
    edge_index = torch.cat(edge_indices, dim=1)
    
    # 创建批次标识
    batch = torch.cat([
        torch.full((g.num_nodes,), i) 
        for i, g in enumerate(graph_list)
    ])

    pos = torch.cat([g.pos for g in graph_list], dim=0)
    
    # 合并其他必要属性（按需添加）
    if hasattr(graph_list[0], 'rxn_exp'):
        rxn_exp = torch.cat([g.rxn_exp for g in graph_list], dim=0)
        return Data(x=x, edge_index=edge_index, slide=batch, pos=pos, rxn_exp=rxn_exp)
    
    return Data(x=x, edge_index=edge_index, pos=pos, slide=batch)


def main():
    
    cfg = TrainingConfig()

    graph_list, met_edge, S, S_tensor, avi_meta, avi_genes = data_preparing(
        # graph_list = '/data/home/liuzhaoyang/data/EcoTIME/EcoTIME_human_tumor_raw_graph_list_10.pth',
        graph_list = '/data/home/liuzhaoyang/data/EcoTIME/EcoTIME_human_tumor_raw_graph_list.pth',
        met_edge = '/data/home/liuzhaoyang/ES/project/MetaSpace/edge_index_without_com_dedup.pkl',
        S_file = '/data/home/liuzhaoyang/ES/project/MetaSpace/S_without_com_trim_dedup.tsv',
        avi_meta = '/data/home/liuzhaoyang/ES/project/MetaSpace/annotated_metabolite_HMDB_ID.txt',
        avi_genes = '/data/home/liuzhaoyang/ES/project/MetaSpace/avi_genes.txt'
    )
    S_meta_mapping = MSpp.build_S_meta_mapping(S, avi_meta, with_compartment=False, verbose=True,
                                              met_md_path='/data/home/liuzhaoyang/ES/project/MetaSpace/met_md.csv')
    n_gene = len(avi_genes)
    n_rxn = S_tensor.shape[1]
    logger = TensorBoardLogger(save_dir=cfg.root_dir, name="Amiya_{}".format(cfg.root_dir))

    model, train_data, val_data, test_data = setup_training(cfg, graph_list, met_edge, n_gene, n_rxn)

    # 初始化Trainer
    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        callbacks=get_callbacks(),
        accumulate_grad_batches=4,  # 梯度累积
        strategy="ddp_find_unused_parameters_true",  # 分布式策略 # ddp_find_unused_parameters_true
        precision=cfg.precision,
        log_every_n_steps=1,
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=cfg.root_dir,
    )

    print(cfg.batch_size)
    
    # 执行训练
    trainer.fit(
        model,
        train_dataloaders=MyDataLoader(
            train_data,
            num_neighbors=[cfg.n_nei] * cfg.nlayer_nei,  # 两阶邻居采样
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last
        ),
        val_dataloaders=MyDataLoader(
            val_data,
            num_neighbors=[cfg.n_nei] * cfg.nlayer_nei,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last
        )
    )

    trainer.test(
        model,
        dataloaders=MyDataLoader(
            test_data,
            num_neighbors=[cfg.n_nei] * cfg.nlayer_nei,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last
        )
    )

    # 保存最终模型
    torch.save(model.amiya.state_dict(), "{}/final_amiya_model.pth".format(cfg.root_dir))

    

if __name__ == '__main__':
    
    torch.set_float32_matmul_precision('high')
    L.seed_everything(42)

    with torch.cuda.amp.autocast():
        main()

