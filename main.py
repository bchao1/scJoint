import torch
import numpy as np
import random
import os
from datetime import datetime

from config import Config
from util.trainingprocess_stage1 import TrainingProcessStage1
from util.trainingprocess_stage3 import TrainingProcessStage3
from util.knn import KNN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main(config): 
    
    # hardware constraint for speed test
    torch.set_num_threads(config.num_threads)
    torch.autograd.set_detect_anomaly(True)
    
    if config.seed != None:
       set_seed(config.seed)
        
    os.environ['OMP_NUM_THREADS'] = '1'
    

    print('Start time: ', datetime.now().strftime('%H:%M:%S'))

    
    # stage1 training
    print('Training start [Stage1]')
    model_stage1= TrainingProcessStage1(config)
    for epoch in range(config.epochs_stage1):
        print('Epoch:', epoch)
        model_stage1.train(epoch)
        
    print('Write embeddings')
    model_stage1.write_embeddings()
    print('Stage 1 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('KNN')
    KNN(config, neighbors = config.knn_neighbors, knn_rna_samples=20000)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    

    # stage3 training
    print('Training start [Stage3]')
    model_stage3 = TrainingProcessStage3(config)
    for epoch in range(config.epochs_stage3):
        print('Epoch:', epoch)
        model_stage3.train(epoch)
        
    print('Write embeddings [Stage3]')
    model_stage3.write_embeddings()
    print('Stage 3 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('KNN stage3')
    KNN(config, neighbors = config.knn_neighbors, knn_rna_samples=20000)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    
if __name__ == "__main__":
    config = Config()
    config.mmd_weight = 1
    config.sim_weight = 0
    config.reduction_weight = 0
    config.normalize_data = False
    config.binarize = True
    config.encoder_layers = 1
    config.encoder_activation = "none"
    main(config)
