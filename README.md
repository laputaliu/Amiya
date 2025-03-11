# Amiya
The coupled GNN model for reconstructing spatial metabolomics from spatial transcriptomics. Project named as CHIMERA, nick name Amiya. Based on pyg.


> ### Notes for Xiantong
> - This project is based on Saturn.
> - Since we haven't packaged the CHIMERA, the file paths in these scripts should be changed according to your own working env.
> - If you have any problem building your working env, you can use my conda env under the following path `/fs/home/liuzhaoyang/biosoft/mambaforge/envs/MS`. But don't directly install any new packages in my env.
> - If you want to test the model, start from a small data size.
> - codes are under `\Amiya`, where `*_pre` for pre-training; `*_fine` for fine-tuning; `_ablation` for ablation experiments. `_L` denotes the training used `Lightning` framework.
>
> Update on 2025/03/11
> - i just update the Amiya on Github to v0.1.7.1 which is the newest version. but only the pre-training part is finished coding and training. the fine-tuning part is on the way (maybe in one week...). if you want a test running, please use at most 2 GPUs and feed with a tiny graph list (e.g. EcoTIME_human_tumor_raw_graph_list_10.pth). If `CUDA out of memory` occurs, try to set a smaller batch size.
