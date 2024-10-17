import numpy as np
import pandas as pd

import torch
import torch_geometric
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn.pool import global_add_pool


class MS_model(torch.nn.Module):
    def __init__(self, num_features, device=torch.device('cuda')):
        '''
        Gene-reaction mappint part:
            just a simple two layer NN for predicting reaction 'exp' from gene exp with g-r relation mask;
            will try more complex model later
            - gr_nchan_in: gene-reaction mapping input dim: num. of input gene
        GraphSAGE part:
            
        '''
        super().__init__()
        # -------------------------------------- #
        # need to add the gene-reaction relation later #
        # this function cannot be used right now #
        # -------------------------------------- #
        
        # self.gr_mapping = Sequential(
        #     Linear(in_features=gr_nchan_in,
        #            out_features=GSAGE_nchan_in),
        #     ReLU()
        # )
        
        # self.batch_size = batch_size
        self.num_features = num_features
        self.device = device

        self.sage_st_1 = GraphSAGE(in_channels=self.num_features, 
                              hidden_channels=1024, 
                              out_channels=self.num_features, 
                              num_layers=2)
                
        self.sage_st_2 = GraphSAGE(in_channels=self.num_features, 
                              hidden_channels=1024, 
                              out_channels=self.num_features, 
                              num_layers=2)
        
        self.mlp_st_1 = Linear(self.num_features,self.num_features)
        self.mlp_st_2 = Linear(self.num_features,self.num_features)
        
        # MLP for connect ST & metabolic graph
        self.mlp_connect = Sequential(
            Linear(self.num_features,self.num_features),
            ReLU(),
            Linear(self.num_features,self.num_features),
            ReLU()
        )
        
        self.sage_meta_1 = GraphSAGE(in_channels=1, 
                  hidden_channels=32, 
                  out_channels=1, 
                  num_layers=2)
        
        self.sage_meta_2 = GraphSAGE(in_channels=1, 
                  hidden_channels=32, 
                  out_channels=1, 
                  num_layers=2)
        
        self.mlp_meta_1 = Linear(self.num_features,self.num_features)
        self.mlp_meta_2 = Linear(self.num_features,self.num_features)
        
        self.mlp_out = Linear(self.num_features,self.num_features)
        
        self.bn_1 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_2 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_3 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_4 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_5 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_6 = torch.nn.BatchNorm1d(num_features=self.num_features)
        
        
    def forward(self, graph, meta_edge_index):
        '''
        '''
        x, edge_index = graph.x.to(self.device), graph.edge_index.to(self.device)
        graph_batch = graph.batch.to(self.device)
        
        # ------------------------------------------------------------------------ #
        # also need to test where to put BN is better (before or after activation) #
        # ------------------------------------------------------------------------ #
        
        
        ##########################
        # embedding spatial info #
        ##########################
        st_embed = self.sage_st_1(x, edge_index)
        st_embed = self.bn_1(st_embed)
        st_embed = F.relu(st_embed)
        st_embed = self.sage_st_2(st_embed, edge_index)
        st_embed = self.bn_2(st_embed)
        st_embed = F.relu(st_embed)
        
        ##############
        # connection #
        ##############
        # sum rxn across the spot
        st_embed = global_add_pool(st_embed, batch=graph_batch)
            # ---------------------------------------- #
            # do we need a BN layer & relu layer here? #
            # ---------------------------------------- #
        # st_embed = self.bn_3(st_embed)
        # st_embed = F.relu(st_embed)
        st_embed = self.mlp_connect(st_embed)
        st_embed = self.bn_3(st_embed)
        st_embed = F.relu(st_embed)
        
        ############################
        # embedding metabolic info #
        ############################
        st_embed = torch.reshape(st_embed, (graph.batch_size, self.num_features, 1))
        # reshape to the metabolic model input (batch_size, n_feature, 1)
        # the dim of node feature in metabolic model is 1 (for now)
        
        st_embed = self.sage_meta_1(st_embed, meta_edge_index)
        st_embed = self.bn_4(st_embed)
        st_embed = F.tanh(st_embed)
        # for metabolic part, negative value is allowed (rxn is reversed), so try tanh here for (-1,1)
        st_embed = self.sage_meta_2(st_embed, meta_edge_index)
        st_embed = self.bn_5(st_embed)
        st_embed = F.tanh(st_embed)
        
        # ##################
        # # output mapping #
        # ##################
        st_embed = torch.reshape(st_embed, (graph.batch_size, self.num_features))
        st_embed = self.mlp_out(st_embed)
        st_embed = self.bn_6(st_embed)
        st_embed = F.tanh(st_embed)
        
        return st_embed

    
    
    
def fba_recon_batch_loss(st_embed, S_tensor, ori_rxn_tensor, 
                         lw_fba=0.5, lw_recon_zero=0.5,
                         device=torch.device('cuda'),
                         return_split=False):
    '''
    cal the loss based on FBA & reconstruction
    
    '''
    st_embed = torch.reshape(st_embed, (st_embed.shape[0], st_embed.shape[1], 1))
    
    ############
    # FBA loss #
    ############
    loss_fba = torch.abs(torch.sum(torch.sum(torch.matmul(S_tensor, st_embed), dim=1)))
    
    ##############
    # recon loss #
    ##############
    # ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (st_embed.shape[0], st_embed.shape[1], 1))
    ori_zero_mask = ori_rxn_tensor == 0
    ori_zero_mask = ori_zero_mask.int().float().to(device)
    ori_nzero_mask = ori_rxn_tensor > 0
    ori_nzero_mask = ori_nzero_mask.int().float().to(device)
    loss_recon_zero = torch.sum(1-F.cosine_similarity(torch.abs(st_embed*ori_zero_mask), ori_rxn_tensor*ori_zero_mask, dim=1))
    loss_recon_nzero = torch.sum(1-F.cosine_similarity(torch.abs(st_embed*ori_nzero_mask), ori_rxn_tensor*ori_nzero_mask, dim=1))
    loss_recon = lw_recon_zero*loss_recon_zero + (1-lw_recon_zero)*loss_recon_nzero
    
    loss = lw_fba*loss_fba + (1-lw_fba)*loss_recon
    
    if return_split:
        return lw_fba*loss_fba, (1-lw_fba)*loss_recon
    else:
        return loss

    
    
def train_batch(dataloader, model, optimizer, meta_edge_index, S_tensor, lw_fba, lw_recon_zero):
    model.train()
    epoch_loss, epoch_loss_fba, epoch_loss_recon = 0, 0, 0
    for n_batch, tmp_graph in enumerate(dataloader):
        optimizer.zero_grad()
        
        # graph = graph.to(device)
        # tmp_x, tmp_edge = graph.x.to(device), graph.edge_index.to(device)
        # out_embed = model(tmp_x, tmp_edge)
        out_embed = model(tmp_graph, meta_edge_index)
        
        # extract the original rxn expression tensor
        ori_rxn_tensor = tmp_graph.x[:tmp_graph.batch_size,]
        ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (out_embed.shape[0], out_embed.shape[1], 1)).to(model.device)

        # cal the loss
        loss_fba, loss_recon = fba_recon_batch_loss(out_embed, S_tensor, ori_rxn_tensor, 
                                                    lw_fba=lw_fba, lw_recon_zero=lw_recon_zero,
                                                    device=model.device,
                                                    return_split=True)
        loss = loss_fba + loss_recon

        epoch_loss_fba += loss_fba.item()
        epoch_loss_recon += loss_recon.item()
        epoch_loss += loss.item()

        loss = loss/tmp_graph.batch_size
        loss.backward()
        optimizer.step()
        
    return epoch_loss, epoch_loss_fba, epoch_loss_recon
                
        
def val_batch(dataloader, model, meta_edge_index, S_tensor, lw_fba, lw_recon_zero):
    model.eval()
    epoch_loss, epoch_loss_fba, epoch_loss_recon = 0, 0, 0
    with torch.no_grad():
        for n_batch, tmp_graph in enumerate(dataloader):

            # graph = graph.to(device)
            # tmp_x, tmp_edge = graph.x.to(device), graph.edge_index.to(device)
            # out_embed = model(tmp_x, tmp_edge)
            out_embed = model(tmp_graph, meta_edge_index)

            # extract the original rxn expression tensor
            ori_rxn_tensor = tmp_graph.x[:tmp_graph.batch_size,]
            ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (out_embed.shape[0], out_embed.shape[1], 1)).to(model.device)

            # cal the loss
            loss_fba, loss_recon = fba_recon_batch_loss(out_embed, S_tensor, ori_rxn_tensor, 
                                                        lw_fba=lw_fba, lw_recon_zero=lw_recon_zero,
                                                        device=model.device,
                                                        return_split=True)
            loss = loss_fba + loss_recon

            epoch_loss_fba += loss_fba.item()
            epoch_loss_recon += loss_recon.item()
            epoch_loss += loss.item()
        
    return epoch_loss, epoch_loss_fba, epoch_loss_recon


def test_batch(dataloader, model, meta_edge_index, S_tensor, lw_fba, lw_recon_zero, S_meta_mapping, avi_meta):
    model.eval()
    epoch_loss, epoch_loss_fba, epoch_loss_recon = 0, 0, 0
    tmp_pearson_corr_list = []
    with torch.no_grad():
        for n_batch, tmp_graph in enumerate(dataloader):

            # graph = graph.to(device)
            # tmp_x, tmp_edge = graph.x.to(device), graph.edge_index.to(device)
            # out_embed = model(tmp_x, tmp_edge)
            out_embed = model(tmp_graph, meta_edge_index)

            # extract the original rxn expression tensor
            ori_rxn_tensor = tmp_graph.x[:tmp_graph.batch_size,]
            ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (out_embed.shape[0], out_embed.shape[1], 1)).to(model.device)

            # cal the loss
            loss_fba, loss_recon = fba_recon_batch_loss(out_embed, S_tensor, ori_rxn_tensor, 
                                                        lw_fba=lw_fba, lw_recon_zero=lw_recon_zero,
                                                        device=model.device,
                                                        return_split=True)
            loss = loss_fba + loss_recon

            epoch_loss_fba += loss_fba.item()
            epoch_loss_recon += loss_recon.item()
            epoch_loss += loss.item()
            
            # cal the correlation
            
            out_metabolite = torch.abs(torch.matmul(S_tensor, torch.reshape(out_embed, (out_embed.shape[0], out_embed.shape[1], 1))))
            out_metabolite = out_metabolite.detach().cpu().numpy()
            out_metabolite = pd.DataFrame(out_metabolite[:,:,0])
            out_metabolite.columns = S_meta_mapping.index
            out_metabolite = pd.concat([out_metabolite, S_meta_mapping.T], axis=0)
            out_metabolite = out_metabolite.dropna(axis=1).T
            
            # merge same metabolite use mean
            out_metabolite['mean'] = out_metabolite.groupby('hmdbID')[0].transform('mean')
            out_metabolite.drop_duplicates(subset=['hmdbID','mean'], inplace=True)
            out_metabolite.index = out_metabolite.loc[:,'hmdbID']
            out_metabolite = out_metabolite.iloc[:,:-2].T

            real_meta_intense = tmp_graph.meta_x[:tmp_graph.batch_size,]
            real_meta_intense = pd.DataFrame(real_meta_intense)
            real_meta_intense.columns = avi_meta
            real_meta_intense = real_meta_intense.loc[:,out_metabolite.columns]

            batch_pearson_corr = out_metabolite.apply(lambda x: pearsonr(x, real_meta_intense.loc[x.name,:])[0], axis=1)
            tmp_pearson_corr_list += list(batch_pearson_corr)
        
    return epoch_loss, epoch_loss_fba, epoch_loss_recon, tmp_pearson_corr_list




###############################
#                             #
# func for recon part refine  #
#                             #
###############################


class MS_model_recon_part(torch.nn.Module):
    def __init__(self, num_features, device=torch.device('cuda')):
        '''
        Gene-reaction mappint part:
            just a simple two layer NN for predicting reaction 'exp' from gene exp with g-r relation mask;
            will try more complex model later
            - gr_nchan_in: gene-reaction mapping input dim: num. of input gene
        GraphSAGE part:
            
        '''
        super().__init__()
        # -------------------------------------- #
        # need to add the gene-reaction relation later #
        # this function cannot be used right now #
        # -------------------------------------- #
        
        # self.gr_mapping = Sequential(
        #     Linear(in_features=gr_nchan_in,
        #            out_features=GSAGE_nchan_in),
        #     ReLU()
        # )
        
        # self.batch_size = batch_size
        self.num_features = num_features
        self.device = device

        self.sage_st_1 = GraphSAGE(in_channels=self.num_features, 
                              hidden_channels=1024, 
                              out_channels=self.num_features, 
                              num_layers=2)
                
        self.sage_st_2 = GraphSAGE(in_channels=self.num_features, 
                              hidden_channels=1024, 
                              out_channels=self.num_features, 
                              num_layers=2)
        
        self.mlp_st_1 = Linear(self.num_features,self.num_features)
        self.mlp_st_2 = Linear(self.num_features,self.num_features)
        
        # MLP for connect ST & metabolic graph
        self.mlp_connect = Sequential(
            Linear(self.num_features,self.num_features),
            ReLU(),
            Linear(self.num_features,self.num_features),
            ReLU()
        )
        
        self.sage_meta_1 = GraphSAGE(in_channels=1, 
                  hidden_channels=32, 
                  out_channels=1, 
                  num_layers=2)
        
        self.sage_meta_2 = GraphSAGE(in_channels=1, 
                  hidden_channels=32, 
                  out_channels=1, 
                  num_layers=2)
        
        self.mlp_meta_1 = Linear(self.num_features,self.num_features)
        self.mlp_meta_2 = Linear(self.num_features,self.num_features)
        
        self.mlp_out = Linear(self.num_features,self.num_features)
        
        self.bn_1 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_2 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_3 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_4 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_5 = torch.nn.BatchNorm1d(num_features=self.num_features)
        self.bn_6 = torch.nn.BatchNorm1d(num_features=self.num_features)
        
        
    def forward(self, graph, meta_edge_index):
        '''
        '''
        x, edge_index = graph.x.to(self.device), graph.edge_index.to(self.device)
        graph_batch = graph.batch.to(self.device)
        
        # ------------------------------------------------------------------------ #
        # also need to test where to put BN is better (before or after activation) #
        # ------------------------------------------------------------------------ #
        
        
        ##########################
        # embedding spatial info #
        ##########################
        st_embed = self.sage_st_1(x, edge_index)
        st_embed = self.bn_1(st_embed)
        st_embed = F.relu(st_embed)
        st_embed = self.sage_st_2(st_embed, edge_index)
        st_embed = self.bn_2(st_embed)
        st_embed = F.relu(st_embed)
        
        ##############
        # connection #
        ##############
        # sum rxn across the spot
        st_embed = global_add_pool(st_embed, batch=graph_batch)
            # ---------------------------------------- #
            # do we need a BN layer & relu layer here? #
            # ---------------------------------------- #
        # st_embed = self.bn_3(st_embed)
        # st_embed = F.relu(st_embed)
        st_embed = self.mlp_connect(st_embed)
        st_embed = self.bn_3(st_embed)
        st_embed = F.relu(st_embed)
        
        ############################
        # embedding metabolic info #
        ############################
        st_embed = torch.reshape(st_embed, (graph.batch_size, self.num_features, 1))
        # reshape to the metabolic model input (batch_size, n_feature, 1)
        # the dim of node feature in metabolic model is 1 (for now)
        
        st_embed = self.sage_meta_1(st_embed, meta_edge_index)
        st_embed = self.bn_4(st_embed)
        st_embed = F.tanh(st_embed)
        # for metabolic part, negative value is allowed (rxn is reversed), so try tanh here for (-1,1)
        st_embed = self.sage_meta_2(st_embed, meta_edge_index)
        st_embed = self.bn_5(st_embed)
        st_embed = F.tanh(st_embed)
        
        # # ##################
        # # # output mapping #
        # # ##################
        st_embed = torch.reshape(st_embed, (graph.batch_size, self.num_features))
        # st_embed = self.mlp_out(st_embed)
        # st_embed = self.bn_6(st_embed)
        # st_embed = F.tanh(st_embed)
        
        return st_embed

    
def train_batch_recon_part(dataloader, model, optimizer, meta_edge_index, S_tensor, lw_fba, lw_recon_zero):
    model.train()
    epoch_loss = 0
    for n_batch, tmp_graph in enumerate(dataloader):
        optimizer.zero_grad()
        
        out_embed = model(tmp_graph, meta_edge_index)
        
        # extract the original rxn expression tensor
        ori_rxn_tensor = tmp_graph.x[:tmp_graph.batch_size,]
        ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (out_embed.shape[0], out_embed.shape[1], 1)).to(model.device)

        # cal the loss
        loss = fba_recon_batch_loss(out_embed, S_tensor, ori_rxn_tensor, 
                                    lw_fba=lw_fba, lw_recon_zero=lw_recon_zero,
                                    device=model.device,
                                    return_split=False)

        epoch_loss += loss.item()

        loss = loss/tmp_graph.batch_size
        loss.backward()
        optimizer.step()
        
    return epoch_loss
                
        
def val_batch_recon_part(dataloader, model, meta_edge_index, S_tensor, lw_fba, lw_recon_zero):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for n_batch, tmp_graph in enumerate(dataloader):

            out_embed = model(tmp_graph, meta_edge_index)

            # extract the original rxn expression tensor
            ori_rxn_tensor = tmp_graph.x[:tmp_graph.batch_size,]
            ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (out_embed.shape[0], out_embed.shape[1], 1)).to(model.device)

            # cal the loss
            loss = fba_recon_batch_loss(out_embed, S_tensor, ori_rxn_tensor, 
                                        lw_fba=lw_fba, lw_recon_zero=lw_recon_zero,
                                        device=model.device,
                                        return_split=False)
            epoch_loss += loss.item()
        
    return epoch_loss


def test_batch_recon_part(dataloader, model, meta_edge_index, S_tensor, lw_fba, lw_recon_zero):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for n_batch, tmp_graph in enumerate(dataloader):

            out_embed = model(tmp_graph, meta_edge_index)

            # extract the original rxn expression tensor
            ori_rxn_tensor = tmp_graph.x[:tmp_graph.batch_size,]
            ori_rxn_tensor = torch.reshape(ori_rxn_tensor, (out_embed.shape[0], out_embed.shape[1], 1)).to(model.device)

            # cal the loss
            loss = fba_recon_batch_loss(out_embed, S_tensor, ori_rxn_tensor, 
                                        lw_fba=lw_fba, lw_recon_zero=lw_recon_zero,
                                        device=model.device,
                                        return_split=False)
            epoch_loss += loss.item()
                    
    return epoch_loss

