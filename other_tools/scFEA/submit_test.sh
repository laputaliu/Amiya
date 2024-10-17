cd ~/biosoft/scFEA-1.1
nohup python ~/biosoft/scFEA-1.1/src/scFEA.py --data_dir ./data --input_dir /mnt/Venus/home/liuzhaoyang/data/GBM_Spatial_MultiOmics/259_T/ --test_file Integrated_ST_count_259_T.tsv --moduleGene_file module_gene_m168.csv --stoichiometry_matrix cmMat_c70_m168.csv --cName_file cName_c70_m168.csv --output_flux_file ~/ES/project/MetaSpace/other_tools/scFEA/259_T_flux.csv --output_balance_file ~/ES/project/MetaSpace/other_tools/scFEA/259_T_balance.csv --sc_imputation True > ~/ES/project/MetaSpace/other_tools/scFEA/259_T_running.log 2>&1 &