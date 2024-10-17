import numpy as np
import pandas as pd
import libsbml
import re
import torch
from torch_geometric.data import Data
from pandarallel import pandarallel


def extract_rxn_meta(meta_model, rxn_boundary_limit=1000,
                    hgcn_db='/fs/home/liuzhaoyang/data/HGNC_custom_db.tsv'):
    '''
    load metabolic model & extract rxns meta info.
    input: metabolic model (.xml) & HGCN gene id database
    output: rxn_meta_dic
    '''
    
    # ---------- #
    # load model #
    # ---------- #
    sbmlDocument = libsbml.readSBMLFromFile(meta_model)
    sbml_model = sbmlDocument.model
    rxn_list = sbml_model.getListOfReactions()

    hgcn_db = pd.read_csv(hgcn_db, sep='\t', index_col=0)
    gr_pattern = re.compile('<html:p>GENE_ASSOCIATION: ([0-9a-zA-Z: ()]*)</html:p>')
    
    rxn_meta_dic = {} # reactant, product & reaction reversible & boundaries & related genes
    # metabolites_dic = {}
    # rxn_metabolites = []
    # all_rxn_ids = []


    for tmp_rxn in rxn_list:
        tmp_rxn_id = tmp_rxn.getId()[2:].replace('_LPAREN_','(').replace('_RPAREN_',')')
        # all_rxn_ids.append(tmp_rxn_id)


        # -------- #
        # rxn type #
        # -------- #
        # tmp_ex_flag = tmp_rxn_id.startswith('EX_')


        # --------------------------------------------------------------------------------------- #
        # just compare whether the react & prod are the same to classify the transport reactions
        # we need to find out a better way later
        # --------------------------------------------------------------------------------------- #


        # tmp_trans_flag = tmp_rxn_id in transport_rxns

        # if tmp_ex_flag:
        #     tmp_rxn_type = 'EX'
        # elif tmp_trans_flag:
        #     tmp_rxn_type = 'TP'
        # else:
        #     tmp_rxn_type = 'RX'

        # if tmp_rxn_id not in rxn_meta_dic:  # do not need this any more, no dupulicated rxn IDs
        rxn_meta_dic.update({
            tmp_rxn_id: {
                # 'rxn_type': tmp_rxn_type
                'rxn_type': 'RX'
            }
        })

        # -------- #
        # reactant #
        # -------- #
        tmp_rea_dic = {}
        for tmp_rea in tmp_rxn.getListOfReactants():
            # tmp_rea_dic 
            # key: reactant name (species)
            # value: Stoichiometry
            tmp_meta_name = tmp_rea.getSpecies()[2:] # trim the prefix of metabolite name
            tmp_rea_dic.update({
                tmp_meta_name: tmp_rea.getStoichiometry() # tmp_rea.getStoichiometry() only have positive value
            })

            # if tmp_meta_name not in rxn_metabolites:
            #     rxn_metabolites.append(tmp_meta_name)
            #     # new metabolite
            #     metabolites_dic.update({
            #         tmp_meta_name: {
            #             tmp_rxn_id: -1 * tmp_rea.getStoichiometry() # add the direction info.
            #         }
            #     })
            # else:
            #     # metabolite already exist
            #     metabolites_dic[tmp_meta_name].update({
            #         tmp_rxn_id: -1 * tmp_rea.getStoichiometry()
            #     })

        rxn_meta_dic[tmp_rxn_id].update({
            'reactants': tmp_rea_dic
        })


        # ------- #
        # product #
        # ------- #
        tmp_prod_dic = {}
        for tmp_prod in tmp_rxn.getListOfProducts():
            # tmp_prod_dic: the same as tmp_rea_dix
            tmp_meta_name = tmp_prod.getSpecies()[2:]
            tmp_prod_dic.update({
                tmp_meta_name: tmp_prod.getStoichiometry()
            })

            # if tmp_meta_name not in rxn_metabolites:
            #     rxn_metabolites.append(tmp_meta_name)
            #     # new metabolite
            #     metabolites_dic.update({
            #         tmp_meta_name: {
            #             tmp_rxn_id: tmp_rea.getStoichiometry()
            #         }
            #     })
            # else:
            #     # metabolite already exist
            #     metabolites_dic[tmp_meta_name].update({
            #         tmp_rxn_id: tmp_rea.getStoichiometry()
            #     })

        rxn_meta_dic[tmp_rxn_id].update({
            'products': tmp_prod_dic
        })


        # -------------- #
        # rxn boundaries #
        # ---------------#
        # the default up and low bound
        for tmp_p in tmp_rxn.getKineticLaw().getListOfParameters():
            if tmp_p.getId() == 'LOWER_BOUND':
                tmp_lb = tmp_p.getValue()
                tmp_lb = -1*rxn_boundary_limit if tmp_lb == -np.inf else tmp_lb
            elif tmp_p.getId() == 'UPPER_BOUND':
                tmp_ub = tmp_p.getValue()
                tmp_ub = rxn_boundary_limit if tmp_ub == np.inf else tmp_ub

        rxn_meta_dic[tmp_rxn_id].update({
            'boundary': {
                'up': tmp_ub,
                'low': tmp_lb
            }
        })


        # ---------- #
        # reversible #
        # ---------- #
        rxn_meta_dic[tmp_rxn_id].update({
            'reversible': tmp_rxn.getReversible()
        })
        
        # ---------------- #
        # gene association #
        # ---------------- #
        tmp_note = tmp_rxn.getNotesString()
        tmp_gid = gr_pattern.search(tmp_note).group(1)
        if len(tmp_gid) == 0:
            # no gene association
            tmp_gsym = []
        else:
            tmp_gid = tmp_gid.replace('(','').replace(')','').replace('and','or').replace('HGNC:HGNC:','HGNC:').split(' or ')
            # there has this kind of note: 'HGNC:30866 or (HGNC:6535 and HGNC:6541) or HGNC:6535'
            # though HGNC:6535 and HGNC:6541 needs to be together, but we only need to use mean or max to cal rxn exp
            # so no diff. for us to just treat them all as 'or'
            # sometimes, notes like 'HGNC:2698 and HGNC:HGNC:987' may also exist
            tmp_gsym = list(hgcn_db.loc[tmp_gid, 'Approved symbol'].values)

        rxn_meta_dic[tmp_rxn_id].update({
            'gene_association': list(set(tmp_gsym))
        })
        
    return rxn_meta_dic


def fill_S(rxn_meta_dic, with_compartment=True, exclude_metabolites=[], 
           trim_zero_meta=True, remove_biomass=True, remove_single_nonzero_rxn=True,
           remove_duplicates=True,
           outdir='./', save_flag=False, save_name='S_with_com.tsv'
          ):
    '''
    build Stoichiometry matrix (S) from rxn_meta_dic, raw as metabolite, col as reaction
    input: 
        - rxn_meta_dic (output of func. extract_rxn_meta)
        - exclude_metabolites: list of metabolite which don't want to exist in the final S (e.g. h2o)
    output: S (pd.DataFrame)
    seems like S without compartment info. basically never be used, 
    so only keep the codes for generating S with compartment
    can also remove some metabolites (e.g. H2O, O2) by exclude_metabolites
    '''
    
    rxn_ids = list(rxn_meta_dic.keys())
    metabolites = []
    
    for tmp_rxn in rxn_ids:
        metabolites += list(rxn_meta_dic[tmp_rxn]['reactants'].keys())
        metabolites += list(rxn_meta_dic[tmp_rxn]['products'].keys())
            
    metabolites = sorted(list(set(metabolites)))
    if not with_compartment:
        metabolites = sorted(list(set([tmp_m[:-2] for tmp_m in metabolites])))
    
    S = pd.DataFrame(np.zeros((len(metabolites), len(rxn_ids))), 
                 columns = rxn_ids, index = metabolites)
    
    for tmp_rxn in rxn_ids:
        for tmp_m in rxn_meta_dic[tmp_rxn]['reactants'].keys():
            tmp_m_s = tmp_m if with_compartment else tmp_m[:-2]
            S.loc[tmp_m_s, tmp_rxn] += -1*rxn_meta_dic[tmp_rxn]['reactants'][tmp_m] # the coefficient of reactant is neg.
        for tmp_m in rxn_meta_dic[tmp_rxn]['products'].keys():
            tmp_m_s = tmp_m if with_compartment else tmp_m[:-2]
            S.loc[tmp_m_s, tmp_rxn] += rxn_meta_dic[tmp_rxn]['products'][tmp_m]
    print('loading S matrix with shape {}'.format(S.shape))
    
    if remove_biomass:
        select_index = [tmp_m.count('biomass') == 0 for tmp_m in list(S.index)]
        select_columns = [tmp_r.count('biomass') == 0 for tmp_r in list(S.columns)]
        S = S.loc[select_index, select_columns]
    
    if len(exclude_metabolites) > 0:
        # the metabolites in exclude_metabolites have no compartment
        S = S.loc[~np.isin([tmp_m[:-2] if with_compartment else tmp_m for tmp_m in list(S.index)], exclude_metabolites),:]
        S = S.loc[:,S.abs().sum(axis=0)!=0]
        
    if remove_single_nonzero_rxn:
        # remove rxns with only one nonzero metabolite
        nonzero_col_rxn = S.apply(lambda x: np.nonzero(x.to_numpy())[0].shape[0], axis=0)
        S = S.loc[:,nonzero_col_rxn>1]
        
    if trim_zero_meta:
        S = S.loc[S.abs().sum(axis=1)!=0,:]
        
    if not with_compartment and remove_duplicates:
        # remove rxns with the same reactants, products & gene associations. (caused by removing compartments)
        tmp_rxn_meta_df = []
        for tmp_r in list(S.columns):
            tmp_dic = rxn_meta_dic[tmp_r]
            tmp_r_str = ','.join(['{}:{}'.format(tmp_k[:-2], tmp_dic['reactants'][tmp_k]) for tmp_k in sorted(list(tmp_dic['reactants'].keys()))])
            tmp_p_str = ','.join(['{}:{}'.format(tmp_k[:-2], tmp_dic['products'][tmp_k]) for tmp_k in sorted(list(tmp_dic['products'].keys()))])
            tmp_g_str = ','.join(sorted(tmp_dic['gene_association']))
            tmp_rxn_meta_df.append([tmp_r_str, tmp_p_str, tmp_g_str])
        tmp_rxn_meta_df = pd.DataFrame(tmp_rxn_meta_df)
        tmp_rxn_meta_df.index = list(S.columns)
        tmp_rxn_meta_df = tmp_rxn_meta_df.sort_index()
        tmp_rxn_meta_df = tmp_rxn_meta_df.drop_duplicates()
        S = S.loc[:,np.isin(list(S.columns), list(tmp_rxn_meta_df.index))]
            
    print('S matrix with shape {} after trimming'.format(S.shape))
        
    if save_flag:
        S.to_csv('{}/{}'.format(outdir,save_name), sep='\t')
        
    return S


def build_S_meta_mapping(S, avi_meta, with_compartment=True, verbose=True,
                         met_md_path='/fs/home/liuzhaoyang/project/MetaSpace/metabolic_model/RECON2_2/met_md.csv'):
    '''
    build the mapping ref. for metabolite name in S and its HMDB ID in MALDI data
    '''
    avi_meta = [tmp_m.replace('HMDB','') for tmp_m in avi_meta]
    ref_id_length = len(avi_meta[0])
    
    # integrate metabolite names in S & metabolite df
    S_meta_meta = pd.read_csv(met_md_path, index_col=0)
    if with_compartment:
        S_meta_meta.index = [tmp_m.replace('[','_').replace(']','') for tmp_m in S_meta_meta.index]
    else:
        S_meta_meta.drop_duplicates(subset=['met_no_compartment'], inplace=True)
        S_meta_meta.index = list(S_meta_meta.loc[:,'met_no_compartment'])
    S_meta_meta = S_meta_meta.loc[np.isin(S_meta_meta.index, S.index),['hmdbID']]
    S_meta_meta.dropna(inplace=True)
    # the length of HMDB ID in annotated metabolites is 11, but in S is 9
    # S_meta_meta.loc[:,'hmdbID'] = S_meta_meta.loc[:,'hmdbID'].apply(lambda x: x.replace('HMDB','').zfill(ref_id_length))
    # S_meta_meta = S_meta_meta.loc[np.isin(S_meta_meta.loc[:,'hmdbID'], avi_meta),:]
    S_meta_meta.loc[:,'hmdbID'] = [tmp_id.replace('HMDB','HMDB00') if tmp_id.replace('HMDB','HMDB00') in avi_meta else np.nan for tmp_id in S_meta_meta.loc[:,'hmdbID']]
    S_meta_meta.dropna(inplace=True)
    
    S_meta_mapping = pd.DataFrame(index=S.index)
    S_meta_mapping = pd.concat([S_meta_mapping, S_meta_meta], axis=1)
    
    if verbose:
        print('{} avi metabolites with mapped HMDB ID'.format(S_meta_mapping.dropna().shape[0]))
    
    return S_meta_mapping


def cal_rxn_exp(gene_exp_df, rxn_meta_dic, avi_rxns=None, merge='max', keep_gr_only=False):
    '''
    calculate rxn exp matrix according to gene exp & gene-reaction associations
    input:
        - gene_exp_df: pd.DataFrame
        - rxn_meta_dic: meta info. dic for rxns, output of extract_rxn_meta
        - avi_rxns: if None all the rxns in this list will be in the final output df, rxn with no gene associated with give 0 exp value
        - merge: in {'max', 'mean'}, how to merge rxn exp which have multi-gene associated
        - keep_gr_only: bool, whether only keep rxn with gene-reaction associations in the final output
    output:
        - the rxn exp df
        
    maybe later will write a multi-processing version. it is slow to use a loop (~2min for 1 slide)
    '''
    
    rxn_exp_df = pd.DataFrame(columns=gene_exp_df.columns)
    for tmp_rxn in rxn_meta_dic.keys():
        tmp_gr = rxn_meta_dic[tmp_rxn]['gene_association']
        if len(tmp_gr) == 0:
            # no gene associated
            tmp_rxn_exp = pd.DataFrame(np.zeros((1, rxn_exp_df.shape[1])), index=[tmp_rxn], columns=rxn_exp_df.columns)
        else:
            tmp_gene_df = gene_exp_df.loc[np.isin(gene_exp_df.index, tmp_gr),:]
            if tmp_gene_df.shape[0] > 0:
                tmp_rxn_exp = tmp_gene_df.max(axis=0) if merge=='max' else tmp_gene_df.mean(axis=0)
                tmp_rxn_exp = pd.DataFrame(tmp_rxn_exp).T
                tmp_rxn_exp.index = [tmp_rxn]
            else:
                # no associated genes existed in gene_exp_df
                tmp_rxn_exp = pd.DataFrame(np.zeros((1, rxn_exp_df.shape[1])), index=[tmp_rxn], columns=rxn_exp_df.columns)
        rxn_exp_df = pd.concat([rxn_exp_df, tmp_rxn_exp], axis=0)
    
    if avi_rxns != None:
        rxn_exp_df = rxn_exp_df.loc[np.isin(rxn_exp_df.index, avi_rxns),:]
        outer_rxns = [tmp_r for tmp_r in avi_rxns if tmp_r not in list(rxn_exp_df.index)]
        if len(outer_rxns) > 0:
            tmp_rxn_exp = pd.DataFrame(np.zeros((len(outer_rxns), rxn_exp_df.shape[1])), index=outer_rxns, columns=rxn_exp_df.columns)
            rxn_exp_df = pd.concat([rxn_exp_df, tmp_rxn_exp], axis=0)
    
    if keep_gr_only:
        rxn_exp_df = rxn_exp_df.loc[rxn_exp_df.sum(axis=1)>0, :]
        
    return rxn_exp_df





###############################
#                             #
#  graph processing functions #
#                             #
###############################





def build_st_graph_tsv(st_exp_tsv, st_coord_tsv, rxn_meta_dic=None, use_rxn_exp=False,
                       xy_cols=['x','y'], validate=True,
                       avi_rxns=None, merge='max', keep_gr_only=False
                      ):
    
    '''
    build a ST graph from the tsv inputs
    input of ST_sct_norm data & its spot coordinates
    'use_rxn_exp' -> trans gene expressions to rxn expressions
    '''
    
    ###################
    # loading ST data #
    ###################

    st_exp = pd.read_csv(st_exp_tsv, sep='\t', index_col=0)
    st_pos = pd.read_csv(st_coord_tsv, sep='\t', index_col=0)
    st_pos = st_pos.loc[list(st_exp.columns), xy_cols]
    # st_pos.columns = ['x', 'y']

    gene_list = list(st_exp.index)
    spot_lsit = list(st_pos.index)
    
    # trans gene exp to rxn exp
    if use_rxn_exp:
        if rxn_meta_dic != None:
            st_exp = cal_rxn_exp(st_exp, rxn_meta_dic, avi_rxns=avi_rxns, merge=merge, keep_gr_only=keep_gr_only)
        else:
            print('rxn_meta_dic is needed')
            return None

    st_exp_tensor = torch.from_numpy(st_exp.T.to_numpy())
    st_exp_tensor = st_exp_tensor.float() # just storge it in float32 to reduce memory usage

    st_pos_tensor = torch.from_numpy(st_pos.to_numpy())
    st_pos_tensor = st_pos_tensor.long()


    ##################
    # build ST graph #
    ##################

    # cal spot-wise Euclidean distance
    tmp_pos = np.array([[complex(st_pos.iloc[i,0], st_pos.iloc[i,1]) for i in range(st_pos.shape[0])]])
    st_dis_matrix = abs(tmp_pos.T-tmp_pos)

    # fill edge_index
    edge_index = []
    for tmp_xi in range(st_dis_matrix.shape[0]):
        tmp_row = st_dis_matrix[tmp_xi]
        tmp_index = np.where(tmp_row == np.min(tmp_row[tmp_row>0]))
        for tmp_yi in tmp_index[0]:
            edge_index.append([tmp_xi,tmp_yi])
            edge_index.append([tmp_yi,tmp_xi])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    st_exp_graph = Data(x=st_exp_tensor, edge_index=edge_index.t().contiguous(), pos=st_pos_tensor)
    # st_exp_graph.to(device)

    # validate graph
    if validate:
        st_exp_graph.validate(raise_on_error=True)
    
    return st_exp_graph


def build_st_graph_metabolite_tsv(st_exp_tsv, st_coord_tsv, st_metabolite_tsv, rxn_meta_dic=None, use_rxn_exp=False,
                       xy_cols=['x','y'], validate=True,
                       avi_rxns=None, merge='max', keep_gr_only=False, avi_meta=None
                      ):
    
    '''
    build a ST graph from the tsv inputs
    input of ST_sct_norm data & its spot coordinates
    'use_rxn_exp' -> trans gene expressions to rxn expressions
    '''
    
    ###################
    # loading ST data #
    ###################

    st_exp = pd.read_csv(st_exp_tsv, sep='\t', index_col=0)
    st_metabolite = pd.read_csv(st_metabolite_tsv, sep='\t', index_col=0)
    st_metabolite = st_metabolite.loc[:,st_exp.columns]
    if avi_meta != None:
        st_metabolite = st_metabolite.loc[np.isin(st_metabolite.index, avi_meta),:]
        st_metabolite = st_metabolite.loc[avi_meta,:]
    st_pos = pd.read_csv(st_coord_tsv, sep='\t', index_col=0)
    st_pos = st_pos.loc[list(st_exp.columns), xy_cols]
    # st_pos.columns = ['x', 'y']

    gene_list = list(st_exp.index)
    spot_lsit = list(st_pos.index)
    
    # trans gene exp to rxn exp
    if use_rxn_exp:
        if rxn_meta_dic != None:
            st_exp = cal_rxn_exp(st_exp, rxn_meta_dic, avi_rxns=avi_rxns, merge=merge, keep_gr_only=keep_gr_only)
        else:
            print('rxn_meta_dic is needed')
            return None

    st_exp_tensor = torch.from_numpy(st_exp.T.to_numpy())
    st_exp_tensor = st_exp_tensor.float() # just storge it in float32 to reduce memory usage
    
    st_meta_tensor = torch.from_numpy(st_metabolite.T.to_numpy())
    st_meta_tensor = st_meta_tensor.float()

    st_pos_tensor = torch.from_numpy(st_pos.to_numpy())
    st_pos_tensor = st_pos_tensor.long()


    ##################
    # build ST graph #
    ##################

    # cal spot-wise Euclidean distance
    tmp_pos = np.array([[complex(st_pos.iloc[i,0], st_pos.iloc[i,1]) for i in range(st_pos.shape[0])]])
    st_dis_matrix = abs(tmp_pos.T-tmp_pos)

    # fill edge_index
    edge_index = []
    for tmp_xi in range(st_dis_matrix.shape[0]):
        tmp_row = st_dis_matrix[tmp_xi]
        tmp_index = np.where(tmp_row == np.min(tmp_row[tmp_row>0]))
        for tmp_yi in tmp_index[0]:
            edge_index.append([tmp_xi,tmp_yi])
            edge_index.append([tmp_yi,tmp_xi])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    st_exp_graph = Data(x=st_exp_tensor, edge_index=edge_index.t().contiguous(), 
                        pos=st_pos_tensor, meta_x=st_meta_tensor)
    # st_exp_graph.to(device)

    # validate graph
    if validate:
        st_exp_graph.validate(raise_on_error=True)
    
    return st_exp_graph


def find_rxn_edge(rxn_col, S, rxn_meta_dic):
    edge_index = []
    
    # find linked rxns
    rxn_S_df = S.multiply(np.array(rxn_col).reshape(-1,1))
    select_rxns = rxn_S_df.apply(lambda x: np.sum(np.array(x) < 0) > 0) # select rxn pairs with connected metabolites
    rxn_S_df = rxn_S_df.loc[:,select_rxns]
    r1_index = list(S.columns).index(rxn_col.name) # the index of target rxn in original S
    r2_index = [list(S.columns).index(tmp_r) for tmp_r in list(rxn_S_df.columns)] # get the index of linked rxn in original S
    
    # check linked rxns directions
    for tmp_i, tmp_r2 in enumerate(r2_index):
        tmp_s = S.iloc[:,[r1_index, tmp_r2]].loc[np.array(rxn_S_df.iloc[:,tmp_i])<0, :]
        # this will only keep the connected metabolites between 2 rxns
        if tmp_s.iloc[0,0] > 0:
            edge_index.append([r1_index, tmp_r2])
            # pos. for r1 & neg. for r2 -> product of r1 is the reactant of r2
        else:
            edge_index.append([tmp_r2, r1_index])
    
    if rxn_meta_dic[rxn_col.name]['reversible']:
        rxn_col = -1*rxn_col
        # for convenient, just copy the codes above...
        rxn_S_df = S.multiply(np.array(rxn_col).reshape(-1,1))
        select_rxns = rxn_S_df.apply(lambda x: np.sum(np.array(x) < 0) > 0) # select rxn pairs with connected metabolites
        rxn_S_df = rxn_S_df.loc[:,select_rxns]
        r1_index = list(S.columns).index(rxn_col.name) # the index of target rxn in original S
        r2_index = [list(S.columns).index(tmp_r) for tmp_r in list(rxn_S_df.columns)] # get the index of linked rxn in original S

        # check linked rxns directions
        for tmp_i, tmp_r2 in enumerate(r2_index):
            tmp_s = S.iloc[:,[r1_index, tmp_r2]].loc[np.array(rxn_S_df.iloc[:,tmp_i])<0, :]
            # this will only keep the connected metabolites between 2 rxns
            if tmp_s.iloc[0,0] > 0:
                edge_index.append([r1_index, tmp_r2])
                # pos. for r1 & neg. for r2 -> product of r1 is the reactant of r2
            else:
                edge_index.append([tmp_r2, r1_index])
                
    return edge_index


def build_metabolic_edge_index(S,rxn_meta_dic,
                               n_workers=1, progress_bar=False):
    if n_workers > 1:
        pandarallel.initialize(progress_bar=progress_bar, nb_workers=n_workers)
        edge_index = S.parallel_apply(find_rxn_edge, args=(S,rxn_meta_dic), axis=0)
    else:
        edge_index = S.apply(find_rxn_edge, args=(S,rxn_meta_dic), axis=0)
    
    return edge_index













