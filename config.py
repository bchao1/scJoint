class Config(object):
	def __init__(self):
	    
		DB = 'db4_control'
        
		if DB =='db4_control':
			# DB info
			self.number_of_class = 7
			self.input_size = 17668
			self.rna_paths = ['data/citeseq_control_rna.npz']
			self.rna_labels = ['data/citeseq_control_cellTypes.txt']		
			self.atac_paths = ['data/asapseq_control_atac.npz']
			self.atac_labels = ['data/asapseq_control_cellTypes.txt'] #Optional. If atac_labels are provided, accuracy after knn would be provided.
			self.rna_protein_paths = ['data/citeseq_control_adt.npz']
			self.atac_protein_paths = ['data/asapseq_control_adt.npz']
			
			# Training config			
			self.batch_size = 256
			self.lr_stage1 = 0.01
			self.lr_stage3 = 0.01
			self.lr_decay_epoch = 20
			self.epochs_stage1 = 20
			self.epochs_stage3 = 20
			self.p = 0.8 # top 80% matches between source and target domain
			self.embedding_size = 64
			self.momentum = 0.9

			self.center_weight = 1
			self.orth_weight = 1
			self.var_weight = 1
			self.reg_weight = 1
			self.mmd_weight = 0 # maximum mean discrepancy
			self.sim_weight = 1
			self.mmd_kernel_num = 5

			self.checkpoint = ''
			self.num_threads = 1
			self.knn_neighbors = 30 # default in paper
			self.seed = 1

			# Additional model configs
			self.encoder_activation = "sigmoid"
			self.encoder_layers = 1
