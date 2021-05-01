if __name__ == "__main__":
    from config import Config
    from util.dataloader_stage1 import *

    config = Config()
    rna_data = Dataloader(True, config.rna_paths[0], config.rna_labels[0], config.rna_protein_paths[0])
    print('rna data:', rna_data.input_size, rna_data.input_size_protein, rna_data.data.shape, rna_data.proteins.shape)
    data = rna_data.data.todense()
    data[data == 0] = np.nan
    rna_mean = np.nanmean(data, 0)
    rna_std = np.nanstd(data, 0)
    print(rna_mean.shape, rna_std.shape)
    data = (data - rna_mean) / rna_std
    data = np.nan_to_num(data)
    print(data)
   
    atac_data = DataloaderWithoutLabel(True, config.atac_paths[0], protien_path = config.atac_protein_paths[0])
    print('atac data:', atac_data.input_size, atac_data.input_size_protein, atac_data.data.shape, atac_data.proteins.shape)
    
    
    #train_rna_loaders, test_rna_loaders, train_atac_loaders, test_atac_loaders, iters = PrepareDataloader(config).getloader()
    #print(len(train_rna_loaders), len(test_atac_loaders))
    
    #print(len(train_rna_loaders[0]), len(train_atac_loaders[0]))
    #print("train iters: ", iters)
