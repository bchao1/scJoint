if __name__ == "__main__":
    from config import Config
    from util.dataloader_stage1 import *

    config = Config()
    rna_data = Dataloader(True, config.rna_paths[0], config.rna_labels[0])
    print('rna data:', rna_data.input_size, rna_data.input_size_protein, rna_data.data.getnnz())
    
    atac_data = DataloaderWithoutLabel(True, config.atac_paths[0])
    print('atac data:', atac_data.input_size, atac_data.input_size_protein, atac_data.data.getnnz())
    
    
    train_rna_loaders, test_rna_loaders, train_atac_loaders, test_atac_loaders, iters = PrepareDataloader(config).getloader()
    print(len(train_rna_loaders), len(test_atac_loaders))
    
    print(len(train_rna_loaders[0]), len(train_atac_loaders[0]))
    print("train iters: ", iters)
