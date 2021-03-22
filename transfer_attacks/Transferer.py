import yaml

# Import Custom Made Victim
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Attack_Metrics import *
from configs.overwrite_config import *
            
class Transferer(): 
    """
    - Collect all the FL NN 
    - Implement transfer attack sweep
    - Hold all the metrics of interest
    """
    
    def __init__(self, filename:str, config_name = None):
        
        # TO IMPLEMENT - Overwrite current file with config_name
        overwrite_config(filename)
        
        # Load config file 
        with open(r'configs/config.yaml') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
            
        self.file = filename
        
        # Matrix to Record Performance (Old Metrics)
        self.orig_acc_transfers = {}
        self.orig_similarities = {}
        self.orig_target_hit = {}
        self.adv_acc_transfers = {}
        self.adv_similarities = {}
        self.adv_target_hit = {}
        
        # Robust-fooled versions
        self.orig_acc_transfers_robust = {}
        self.orig_similarities_robust = {}
        
        self.orig_acc_transfers_adv = {}
        self.orig_similarities_adv = {}
        
        # Attack success indices
        self.adv_indices = {}
        self.robust_indices = {}
        
        # Matrix to Record Performance (New Metrics - theoretical)
        self.metric_variance = None # Single value
        self.metric_alignment = {} # Dict - key is victim NN id
        self.metric_ingrad = {} # Dict - key is victim NN id
        
        self.metric_alignment_robust = {} # Dict - key is victim NN id
        self.metric_ingrad_robust = {} # Dict - key is victim NN id
        
        self.metric_alignment_adv = {} # Dict - key is victim NN id
        self.metric_ingrad_adv = {} # Dict - key is victim NN id
        
        # Attack Params
        self.ifsgm_params = IFSGM_Params()
        self.cw_params = CW_Params()
        
        # Other Params
        self.advNN_idx = None # int
        self.advNN = None # pytorch nn
        self.victim_idxs = None # List of ints
        self.victims = None # dict of pytorch nn
        
        # Recorded Data Points
        self.x_orig = None
        self.y_orig = None
        self.y_true = None
        self.x_adv = None
        self.y_adv = None

        
    def generate_advNN(self, client_idx):
        """
        Select specific client to load neural network to 
        Load the data for that client
        Lod the weights for that client
        This is the client that will generate perturbations
        """
        
        # Import Data Loader for this FL set
        file_indices = [i for i in range(self.config['num_sets'])]
        client_slice = len(file_indices)//self.config['num_clients']
        
        # Import the loader for this dataset only
        self.loader = Dataloader(file_indices,[client_idx*(client_slice),min((client_idx+1)*(client_slice),35)])  
        self.loader.load_training_dataset()
        self.loader.load_testing_dataset()
        
        self.advNN_idx = client_idx
        self.advNN = load_FLNN(idx=client_idx, direc=self.file, loader=self.loader)
        
        return
    
    def generate_xadv(self, atk_type = "IFSGM", mode='test'):
        """
        Generate perturbed images
        atk_type - "IFSGM" or "CW"
        """
        
        if (atk_type == "IFSGM") or (atk_type == "ifsgm"): 
            self.advNN.i_fgsm(self.ifsgm_params, mode=mode)
        elif (atk_type == "CW") or (atk_type == "cw"):
            self.advNN.CW_attack(self.cw_params,mode=mode)
        else:
            print("Attak type unidentified -- Running IFSGM")
            self.advNN.i_fgsm(self.ifsgm_params, mode=mode)
        
        # Record relevant tensors
        self.x_orig = self.advNN.x_orig
        self.y_orig = self.advNN.output_orig
        self.y_true = self.advNN.y_orig
        self.x_adv = self.advNN.x_adv
        self.y_adv = self.advNN.output_adv
    
    def generate_victims(self, client_idxs):
        """
        Load the pre-trained other clients in the system
        """
        
        self.victim_idxs = client_idxs
        self.victims = {}
    
        for i in self.victim_idxs:
            self.victims[i] = load_FLNN(idx=i, direc=self.file, loader=None)
    
    def send_to_victims(self, client_idxs):
        """
        Send pre-generated adversarial perturbations 
        client_idxs - list of indices of clients we want to attack (just victims)
        
        Then record the attack success stats accordingly
        """
        
        for i in client_idxs:
          
            self.victims[i].forward_transfer(self.x_orig,self.x_adv,
                                         self.y_orig,self.y_adv,
                                         self.y_true, self.ifsgm_params.target, 
                                         print_info=False)

            # Record Performance
            self.orig_acc_transfers_robust[i] = self.victims[i].orig_test_acc_robust
            self.orig_similarities_robust[i] = self.victims[i].orig_output_sim_robust

            self.orig_acc_transfers_adv[i] = self.victims[i].orig_test_acc_adv
            self.orig_similarities_adv[i] = self.victims[i].orig_output_sim_adv
                
            # Record Performance
            self.orig_acc_transfers[i] = self.victims[i].orig_test_acc
            self.orig_similarities[i] = self.victims[i].orig_output_sim
                
            self.orig_target_hit[i] = self.victims[i].orig_target_achieve

            self.adv_acc_transfers[i] = self.victims[i].adv_test_acc
            self.adv_similarities[i] = self.victims[i].adv_output_sim
            self.adv_target_hit[i] = self.victims[i].adv_target_achieve

            # Record indices
            self.adv_indices[i] = self.victims[i].adv_indices
            self.robust_indices[i] = self.victims[i].robust_indices
            
    def check_empirical_metrics(self, orig_flag = True, batch_size = 1000):
        """
        Computes the following for the following models:
        - Size of input gradient - across data distribution across all victim NN
        - Gradient Alignment - Between the surrogate and each of the victim NN
        - Variance of loss - Just for the surrogate
        
        - Orig flag false uses new fresh data as inputs instead of xorig and yorig
          (used to attack victims)
        """
        
        # Load a Sample of data from the datalaoder
        if orig_flag: # Split between fooled and not fooled

            self.metric_variance = calcNN_variance(self.advNN, self.x_orig, self.y_orig)
            # For robust data
            for i in range(len(self.victims)):
                data_x = self.x_orig[self.robust_indices[i]]
                data_y = self.y_orig[self.robust_indices[i]]
                
                if (data_y.numel()):
                    self.metric_alignment_robust[i] = calcNN_alignment(self.advNN, self.victims[i], data_x, data_y) 
                    self.metric_ingrad_robust[i] = calcNN_ingrad(self.victims[i],data_x,data_y)
                else:
                    self.metric_alignment_robust[i] = 0
                    self.metric_ingrad_robust[i] = 0
                
                data_x = self.x_orig[self.adv_indices[i]]
                data_y = self.y_orig[self.adv_indices[i]]
                
                if (data_y.numel()):
                    self.metric_alignment_adv[i] = calcNN_alignment(self.advNN, self.victims[i], data_x, data_y) 
                    self.metric_ingrad_adv[i] = calcNN_ingrad(self.victims[i],data_x,data_y) 
                else:
                    self.metric_alignment_adv[i] = 0
                    self.metric_ingrad_adv[i] = 0

                self.metric_alignment[i] = calcNN_alignment(self.advNN, self.victims[i], data_x, data_y) 
                self.metric_ingrad[i] = calcNN_ingrad(self.victims[i],data_x,data_y)
        
        else: # Take measurements across all points
            image_data = self.advNN.dataloader.load_batch(batch_size)
            data_x  = torch.Tensor(image_data['input']).reshape(batch_size,1,28,28)
            data_y = torch.Tensor(image_data['label']).type(torch.LongTensor)

            if torch.cuda.is_available():
                data_y = data_y.cuda()

            self.metric_variance = calcNN_variance(self.advNN, data_x, data_y)
            for i in range(len(self.victims)):
                self.metric_alignment[i] = calcNN_alignment(self.advNN, self.victims[i], data_x, data_y) 
                self.metric_ingrad[i] = calcNN_ingrad(self.victims[i],data_x,data_y)
            
            