from femnist_dataloader import Dataloader
from full_cnn import Full_CNN
from cnn_head import CNN_Head
from cnn_neck import CNN_Neck
from utilities import freeze_layers
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

exp_name = input('Please Enter Experiment Name : ')
mode 		= 'cuda'#input('Please Enter Mode : ')
loader = Dataloader()    
loader.load_training_dataset()
loader.load_testing_dataset()

full_network        = Full_CNN(mode)
cnn_neck        = CNN_Neck(mode)
cnn_head        = CNN_Head(mode)

experiment_losses = []
test_score = []

batch_size = 256
neck_op = torch.empty(5, 7, requires_grad=True).type(torch.cuda.FloatTensor)

for i in range(5000):
    
    cnn_head = freeze_layers(cnn_head,[0,1])
    
    image_data = loader.load_batch(batch_size=batch_size)
    
    ip          = torch.Tensor(image_data['input']).reshape(batch_size,1,28,28)
    if(mode=='cuda'):
        y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()
    else:
        y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor)
		
    neck_op = cnn_neck.forward(ip)       
    
    head_op = cnn_head.forward(neck_op)
    
    loss = full_network.loss_critereon(head_op,y_target)
    experiment_losses.append(loss.cpu().detach().numpy())
    
    cnn_head.optimizer.zero_grad() 
    cnn_neck.optimizer.zero_grad()
    loss.backward()
    cnn_head.optimizer.step()	
    cnn_neck.optimizer.step()
    
    if(i%100==0):
        correct = 0

        for j in range(loader.testing_size):
			
            image_data = loader.load_batch(batch_size=1,mode='test')
			
            ip          = torch.Tensor(image_data['input']).reshape(1,1,28,28)
            if(mode=='cuda'):
                y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()
            else:
                y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()

            neck_op = cnn_neck.forward(ip)           
            head_op = cnn_head.forward(neck_op).cpu().detach().numpy()[0]
            estimate    = int(np.argmax(head_op))
			
            correct+=int(y_target==estimate)
            
        test_score.append(correct/loader.testing_size)
        
        print("Iteration ",i," | Loss : ",loss.cpu().detach().numpy())
        print("Test Accuracy : ",correct/loader.testing_size)
        print()
    
    # Gradient processes if any
    x = full_network.get_activations_gradient()
    
print('Testing Accuracy : ',correct/loader.testing_size)

plt.title("Full CNN Losses")
plt.plot(experiment_losses)
plt.grid()
plt.savefig(exp_name+'_fullnetwork')
plt.show()
#plt.clf()

plt.title("Testing Accuracy")
plt.plot(test_score)
plt.grid()
plt.savefig(exp_name+'_testscores')
plt.show()
#plt.clf()
