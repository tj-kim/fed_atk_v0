import yaml

# Import Custom Made Victim
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *


class Transferer(): 
    """
    - Collect all the FL NN 
    - Implement transfer attack sweep
    - Hold all the metrics of interest
    """
    
    def __init__(self, filename:str, config_name = None):
        
        # TO IMPLEMENT - Overwrite current file with config_name
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
        
        # Matrix to Record Performance (New Metrics - theoretical)
        
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
    
    def generate_xadv(self, atk_type = "IFSGM"):
        """
        Generate perturbed images
        atk_type - "IFSGM" or "CW"
        """
        
        if (atk_type == "IFSGM") or (atk_type == "ifsgm"): 
            self.advNN.i_fgsm(self.ifsgm_params)
        elif (atk_type == "CW") or (atk_type == "cw"):
            self.advNN.CW_attack(self.cw_params)
        else:
            print("Attak type unidentified -- Running IFSGM")
            self.advNN.i_fgsm(self.ifsgm_params)
        
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
            self.orig_acc_transfers[i] = self.victims[i].orig_test_acc
            self.orig_similarities[i] = self.victims[i].orig_output_sim
            self.orig_target_hit[i] = self.victims[i].orig_target_achieve

            self.adv_acc_transfers[i] = self.victims[i].adv_test_acc
            self.adv_similarities[i] = self.victims[i].adv_output_sim
            self.adv_target_hit[i] = self.victims[i].adv_target_achieve
                    
