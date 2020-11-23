# IMPORTING LIBARIES #
# tabular data manipulation
import pandas as pd
# array manipulation
import numpy as np
# visualizations
import matplotlib.pyplot as plt
# operating system status
import os
# for resizing images
from skimage import io, transform
# library for adam optimizer
import torch.optim as optim
# DataLoader is a Python iterable over a dataset
from torch.utils.data import Dataset, DataLoader
# for grabbing a random set of images
from torch.utils.data.sampler import SubsetRandomSampler
# for confusion matrix (model evaluation)
import sklearn.metrics as mt
# for PyTorch Neural Networks
import torch
import torch.nn.functional as F
import torch.nn as nn

# PARAMETERS #

file_path = 'Data Sets/CelebA/annot/list_attr_celeba.txt'
images_path = 'Data Sets/CelebA/img_align_celeba/'
load_net = 0  # Testing = 1
use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
shape = (12, 12)
ip_shape = (150, 150)

batch_size = 256 # added myself
learning_rate = 1e-3
epochs = 10
d = 20

train_ratio = .7
val_ratio = .1
test_ratio = 1-train_ratio-val_ratio

# DATA SET CONFIG #

class Male_Female_dataset(Dataset):

    def __init__(self, root_dir, shape, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.shape = shape
        self.images_names = os.listdir(images_path)

    def __len__(self):
        return len(os.listdir(images_path))

    def __getitem__(self, idx):
        img_n = self.images_names[idx]
        img1_name = os.path.join(self.root_dir, (img_n))
        image1 = io.imread(img1_name)
        image1 = transform.resize(image1, (150, 150))
        col = cele_attrib[cele_attrib.ImgId == img_n]

        t = torch.rand(1)
        if t > 0.5:
            h = torch.randint(0, len(self), (1, 1))
            img2_name = os.path.join(self.root_dir, (self.images_names[h]))
            image2 = io.imread(img2_name)
            image2 = transform.resize(image2, self.shape)
            image2 = torch.Tensor.float(torch.from_numpy(image2))
            image2 = (torch.Tensor.permute(image2, (2, 0, 1)))
            image2 = image2/256.0
            image1 = torch.Tensor.float(torch.from_numpy(image1))
            image1 = (torch.Tensor.permute(image1, (2, 0, 1)))
            image1 = image1/256.0
            annot = 0
            gender = col['Male'].values[0]

        else:
            image2 = image1
            image2 = transform.resize(image2, self.shape)
            image2 = torch.Tensor.float(torch.from_numpy(image2))
            image2 = (torch.Tensor.permute(image2, (2, 0, 1)))
            image2 = image2/256.0
            image1 = torch.Tensor.float(torch.from_numpy(image1))
            image1 = (torch.Tensor.permute(image1, (2, 0, 1)))
            image1 = image1/256.0
            annot = 1
            gender = col['Male'].values[0]
        # annot = annot.astype('float').reshape(-1,2)
        sample = {'image1': image1, 'image2': image2,
                  'same': annot, 'gender': gender}
        return sample


columns = ['ImgId', '5_o_Clock_Shadow', ' Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
           'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
# lfw = cele_attrib.set_index('SUBJECT_ID')
cele_attrib = pd.read_csv(file_path, delimiter="\s+", names=columns)
lfw = cele_attrib

dataset = Male_Female_dataset(images_path, shape)

# DATA LOADER #

# create randomize numpy array with index numbers ranging from 0 to 5 less than the size of the data set
# not sure why it is -5?
index = np.random.permutation(len(dataset)-5)
# number of entries in the training data set
train_data_length = int(train_ratio*len(index))
# number of entries in the validation data set
val_data_length = int(val_ratio*len(index))
# number of entries in the testing data set
test_data_length = int(test_ratio*len(index))
# train data indexes correspond to index 0 of index array to train data length 
train_index = index[:train_data_length]
# validation data indexes correspond to the index values of train data length to validation data length   
val_index = index[train_data_length:(train_data_length+val_data_length)]
# test data indexes correspond to the index values of  validation data length to test data length   
test_index = index[train_data_length+val_data_length:]


train_loader = DataLoader(dataset,batch_size=100,sampler = SubsetRandomSampler(train_index))
val_loader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(val_index))
test_loader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(test_index))

# DISPLAY IMAGES FROM DATASET #

'''
len_attrib = len(cele_attrib)
# Select random images form celeba dataset
rnd_set = np.random.permutation(len_attrib)[0:5]
for i in rnd_set:
     idx = ("{:06d}.jpg".format(i))
     img_path = images_path+idx
     img = plt.imread(img_path)
     print(idx)
     print(cele_attrib['ImgId'][i-1])  
     print(cele_attrib['Male'][i-1]) # i-1 because indexing starts at 0
     plt.imshow(img)   
     plt.show()
'''

'''
def display_images(in_, out, n=1, label=None, count=False):
    for N in range(n):
        if in_ is not None:
            in_pic = in_.data.cpu().view(-1, 150, 150)
            plt.figure(figsize=(18, 4))
            plt.suptitle(label + ' – real test data / reconstructions', color='w', fontsize=16)
            for i in range(4):
                plt.subplot(1,4,i+1)
                plt.imshow(in_pic[i+4*N])
                plt.axis('off')
        out_pic = out.data.cpu().view(-1, 150, 150)
        plt.figure(figsize=(18, 6))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(out_pic[i+4*N])
            plt.axis('off')
            if count: plt.title(str(4 * N + i), color='w')
'''

# CHECKPOINTS #

def save_checkpoint(state, filename='Checkpoints/checkpoint_0.001.pth.tar'):
    torch.save(state, filename)
# def save_checkpoint_A(state, filename='/home/jamal/Downloads/additional_files/checkpoint_A_0.001.pth.tar'):

def save_checkpoint_A(state, filename='Checkpoints/checkpoint_A_0.001.pth.tar'):
    torch.save(state, filename)

# VAE NEURAL NETWORK ##

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(150*150, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 150*150),
            nn.Sigmoid(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(12*12, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 12*12), # rconstructed image
            nn.Sigmoid(),
        )

        self.ip1 = nn.Linear(1712,128)   # change the first parameter in case you change the size of the small image
        self.ip2 = nn.Linear(128,2)
        self.ip3 = nn.Linear(1712,128,False)   # change the first parameter in case you change the size of the small image
        self.ip4 = nn.Linear(128,2,False)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x1, x2):
        mu_logvar1 = self.encoder1(x1.view(-1, 150*150)).view(-1, 2, d)
        mu1 = mu_logvar1[:, 0, :]
        logvar1 = mu_logvar1[:, 1, :]
        z1 = self.reparameterise(mu1, logvar1)
        recnstrct1 = self.decoder1(z1)

        mu_logvar2 = self.encoder2(x2.view(-1, 12*12)).view(-1, 2, d)
        mu2 = mu_logvar2[:, 0, :]
        logvar2 = mu_logvar2[:, 1, :]
        z2 = self.reparameterise(mu2, logvar2)
        recnstrct2 = self.decoder2(z2)

        print(z1.shape)
        print(recnstrct1.shape)
        print(z2.shape)
        print(recnstrct2.shape)

        z = z1-z2
        
        # recnstrct1 = recnstrct1.view(-1,512) # originally 512
        # recnstrct2 = recnstrct2.view(-1,1200)

        # cat = torch.cat((recnstrct1,recnstrct2),1)

        facesame = self.ip1(z)
        facesame = F.relu(facesame)
        facesame = self.ip2(facesame)

        return facesame, z, recnstrct1, mu1, logvar1, recnstrct2, mu2, logvar2


# adversarial neural network (male/female identifier)
class Net_A(nn.Module):
    
    def __init__(self):
        super(Net_A,self).__init__()
        
        self.ip3 = nn.Linear(
            # input = size of flattened image
            1712,
            # output (heavily condensing)
            128,
            # bias set to False, the layer will not learn an additive bias (Default: True)
            False)   # change the first parameter in case you change the size of the small image
        
        # Relu between layers    
        
        self.ip4 = nn.Linear(128,
                             # output corresponds to two nodes: one for female and the other for male ?
                             2,
                             # bias set to False, the layer will not learn an additive bias (Default: True)
                             False)
        
    
    def forward(self,x):
        # takes only the larger image as an input to test in the model
        x2 = self.ip3(x)
        x2 = F.relu(x2)
        x2 = self.ip4(x2)
        # x2 = x2.mul(-1)
        # x = F.relu(x)
        # x = F.softmax(x,1)
        return x2


# LOADING / INITIALIZING NEURAL NETWORK #

if load_net:
    net = VAE().to(device)
    net_A = Net_A().to(device)
    
    # checkpoint = torch.load('/home/jamal/Downloads/baseline young celeba/checkpoint_baseline_celeb.pth.tar')
    # checkpoint_A = torch.load('/home/jamal/Downloads/baseline young celeba/checkpoint_A_baseline_celeb.pth.tar')

    checkpoint = torch.load('Checkpoints/checkpoint_baseline_celeb.pth.tar')
    checkpoint_A = torch.load('Checkpoints/checkpoint_A_baseline_celeb.pth.tar')

    net.load_state_dict ( checkpoint['state_dict'])
    optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005)
    optimizer.load_state_dict = checkpoint['optimizer']
    
    net_A.load_state_dict ( checkpoint_A['state_dict'])
    optimizer_A = optim.Adam(net_A.parameters(),lr = 0.001, weight_decay = 0.0005)
    optimizer_A.load_state_dict = checkpoint_A['optimizer']
else:
    net = VAE().to(device)
    net_A = Net_A().to(device)
    optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005) # Setting the optimiser
    optimizer_A = optim.Adam(net_A.parameters(),lr = 0.001, weight_decay = 0.0005)
    checkpoint= {'epoch':0}

# model = VAE().to(device)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar, shape):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, shape), reduction='sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return BCE + KLD


# error of face matching
err_same = []
# error of gender matching
err_gender = []

# accuracy of face matching
acc1 = []
# accuracy of gender matching
acc2 = []

# minimization criterion of face matching NN
criterion = nn.CrossEntropyLoss()
# minimization criterion of gender matching NN
criterion2 = nn.CrossEntropyLoss()

# not sure what t is ? it's only use seems to be commented out
t = -1

# run for 5 epochs
for ep in range(checkpoint['epoch'],5):
    for i,data in enumerate(train_loader):
        
        # Zeros descent gradient to help generalize after each batch
        optimizer.zero_grad()
        optimizer_A.zero_grad()
        
        # input1 = high res true image
        # input2 = low res true or false image
        # label = name corresponding to true image path
        # gender = male or female label
        input1, input2 , label, gender = data.items()
        # run on cuda gpu if available
        input1, input2 = input1[1].to(device), input2[1].to(device)
        label1 = label[1].to(device)
        gender = gender[1].to(device)
        # not sure why it is (gender+1)/2? 
        gender = (gender+1)//2
        #label = torch.Tensor.long(label[1])
        
        # feed input images 1 and 2 into neural network
        output = net(input1,input2)
        # feed output of main face matching NN to gender matching NN
        output_A = net_A(output[1])

        # ===================backward====================

        # use cross entropy loss as minimization criterion for both main an adv. NN
        # use true image as label
        loss1 = criterion (output[0],label1)
        #loss1.backward(retain_graph=True)
        # optimizer.step()

        # use gender as label
        loss2 = criterion2 (output_A,gender)
        
        # backwards propagation 
        # not sure why retain graph ? In essence, it will retain any necessary information to calculate a certain variable, so that we can do backward pass on it.
        loss2.backward(retain_graph=True)
        # gender matching NN gradient step
        optimizer_A.step()

        # reconstruction loss 
        # facesame, cat, recnstrct1, mu1, logvar1, recnstrct2, mu2, logvar2
        # NOT SURE IF X is the same ??????????????????????????????
        reconstruct1loss = loss_function(output[2], net, output[3], output[4], 150*150)
        reconstruct2loss = loss_function(output[5], net, output[6], output[7], 12*12)

        reconstructionloss = reconstruct1loss + reconstruct2loss

        # formula: L = Ly - wLd
        loss1 = loss1 + reconstructionloss - loss2.mul(1)  #new line 
        #loss2 = loss2.abs()
        # t = t*-1

        # backwards propagation 
        loss1.backward()
        # face matching NN gradient step
        optimizer.step()
        # adding loss as an entry to face matching list
        err_same.append(loss1.item())
        # adding loss as an entry to face matching list
        err_gender.append(loss2.item())

        # if 5th iteration, print the loss, iteration #, and epoch for each of the NN
        if (i%5==0):
            print (loss1,i,ep)
            print (loss2,i,ep)
            #print (output[1])

        # if 50th iteration, check accuracy of model          
        if (i%25==0):
            # create a numpy array from the indexes from the validation array
            val_index1 = np.random.permutation(val_index)[:100]
            # test for accuracy using 30 images from validation data 
            val_dataloader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(val_index1))
            # make validation data iterable
            val_iter = iter(val_dataloader)

            # counter variables
            total = 0
            correct1 = 0
            correct2 = 0
            # literally just going through and checking if the model predicts the true value (same face / which gender) correctly
            for j,dataj in enumerate(val_dataloader):
                input1j, input2j, labelj, gender = dataj.items()
                input1j, input2j = input1j[1].to(device), input2j[1].to(device)
                
                labelj = labelj[1].to(device)
                gender = gender[1].to(device)
                gender = (gender+1)/2
                output = net(input1j,input2j)
                output_A = net_A(output[1])
                _,predicted1 = torch.max(output[0].data,1)
                _,predicted2 = torch.max(output_A.data,1)
                total +=labelj.size(0)
                correct1 += (predicted1 == labelj).sum().item()
                correct2 += (predicted2 == gender).sum().item()
            # prints out percent accurate for both NN
            print('Accuracy_LH: %d %%'%(100*correct1/total))
            print('Accuracy_MF: %d %%'%(100*correct2/total))
            
            # record the percent accurate for both NN into respective lists
            acc1.append(100*correct1/total) #low high
            acc2.append(100*correct2/total) # male female

    # outside of inner loop
    # saves for every epoch

    # save the checkpoint to be loaded for next use
    save_checkpoint({
            # increase the track of the current epoch #
            'epoch': ep + 1,
            # save Python dictionary object that maps each layer to its parameter tensor.
            'state_dict': net.state_dict(),
            # update optimizer conditions (because it may not be zeroed)
            'optimizer' : optimizer.state_dict(),
        })
    save_checkpoint_A({
            'epoch': ep + 1,
            'state_dict': net_A.state_dict(),
            
            'optimizer' : optimizer_A.state_dict(),
        })



'''
codes = dict(μ=list(), logσ2=list(), y=list())
for epoch in range(0, epochs + 1):
    # Training
    if epoch > 0:  # test untrained net first
        net.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar = VAE(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
    # Testing

    means, logvars, labels = list(), list(), list()
    with torch.no_grad():
        net.eval() 
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar = VAE(x)
            test_loss += loss_function(x_hat, x, mu, logvar).item()
            # =====================log=======================
            means.append(mu.detach())
            logvars.append(logvar.detach())
            labels.append(y.detach())
    # ===================log========================
    codes['μ'].append(torch.cat(means))
    codes['logσ2'].append(torch.cat(logvars))
    codes['y'].append(torch.cat(labels))
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    display_images(x, x_hat, 1, f'Epoch {epoch}')
'''