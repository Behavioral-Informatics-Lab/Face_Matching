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
from torch.autograd import Variable

# PARAMETERS #

file_path = 'Data Sets/CelebA/annot/list_attr_celeba.txt'
images_path = 'Data Sets/CelebA/img_align_celeba/'
load_net = False
test_net = False
use_cuda = torch.cuda.is_available()  # use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(True) # debugging tool
dim_LR = 128
shape = (dim_LR, dim_LR)
dim_HR = 128
ip_shape = (dim_HR, dim_HR)
batch_size = 256 # + size, + speed, - quality
learning_rate = 1e-3
epochs = 5
latent_size = 512 
train_ratio = .7
val_ratio = .3
test_ratio = 1-train_ratio-val_ratio

# DATA SET CONFIG #


class get_transformed_celeba(Dataset):

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
        image1 = transform.resize(image1, (dim_HR, dim_HR))
        col = cele_attrib[cele_attrib.ImgId == img_n]

        t = torch.rand(1)
        if t > 0.5: # 50% chance of same face or different
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
            pale = col['Pale_Skin'].values[0]
            young = col['Young'].values[0]

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
            pale = col['Pale_Skin'].values[0]
            young = col['Young'].values[0]

        # annot = annot.astype('float').reshape(-1,2)
        sample = {'image1': image1, 'image2': image2,
                  'same': annot, 'gender': gender,
                  'pale': pale, 'young': young}
        return sample


columns = ['ImgId', '5_o_Clock_Shadow', ' Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
           'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
cele_attrib = pd.read_csv(file_path, delimiter="\s+", names=columns)

dataset = get_transformed_celeba(images_path, shape)

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

train_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=SubsetRandomSampler(train_index))
# val_loader = DataLoader(dataset, batch_size=batch_size,
#                         sampler=SubsetRandomSampler(val_index))
test_loader = DataLoader(dataset, batch_size=batch_size,
                         sampler=SubsetRandomSampler(test_index))

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

def display_img(img, dim):
    in_pic = img.data.cpu().view(-1, dim, dim)
    plt.figure(figsize=(5, 5))
    plt.imshow(in_pic[0])
    plt.axis('off')
    plt.show()

## NETWORK ARCHITECTURES ##

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc # numbrt of channels 
        self.ngf = ngf # ndf = number of filters in the discriminator
        self.ndf = ndf # ngf = number of filters in the generator
        self.latent_variable_size = latent_variable_size # size of latent variable

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar, z


class Net_Attr(nn.Module):

    def __init__(self):
        super(Net_Attr, self).__init__()

        self.ip1 = nn.Linear(
            # input = size of laten vector
            latent_size,
            # output (heavily condensing)
            latent_size//4,
            # bias set to False, the layer will not learn an additive bias (Default: True)
            False)   # change the first parameter in case you change the size of the small image

        # Relu between layers

        self.ip2 = nn.Linear(latent_size//4,
                             # output corresponds to two nodes: one for female and the other for male
                             2,
                             # bias set to False, the layer will not learn an additive bias (Default: True)
                             False)

    def forward(self, z):
        # takes z for predicting attribute or not
        y = self.ip1(z)
        y = F.relu(y)
        y = self.ip2(y)
        # x2 = x2.mul(-1)
        # x = F.relu(x)
        # x = F.softmax(x,1)
        return y


# CHECKPOINTS #

def save_checkpoint(state, filename='Checkpoints/checkpoint_0.001.pth.tar'):
    torch.save(state, filename)
# def save_checkpoint_A(state, filename='/home/jamal/Downloads/additional_files/checkpoint_A_0.001.pth.tar'):

def save_checkpoint_A(state, filename='Checkpoints/checkpoint_A_0.001.pth.tar'):
    torch.save(state, filename)

# LOADING / INITIALIZING NEURAL NETWORK #

if load_net: # MUST GO BACK AND UPDATE
    vae1 = VAE().to(device)
    vae2 = VAE().to(device)

    # checkpoint = torch.load('/home/jamal/Downloads/baseline young celeba/checkpoint_baseline_celeb.pth.tar')
    # checkpoint_A = torch.load('/home/jamal/Downloads/baseline young celeba/checkpoint_A_baseline_celeb.pth.tar')

    checkpoint = torch.load('Checkpoints/checkpoint_baseline_celeb.pth.tar')
    checkpoint_A = torch.load(
        'Checkpoints/checkpoint_A_baseline_celeb.pth.tar')

    net.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer.load_state_dict = checkpoint['optimizer']

    net_A.load_state_dict(checkpoint_A['state_dict'])
    optimizer_A = optim.Adam(net_A.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer_A.load_state_dict = checkpoint_A['optimizer']
else:
    # NETWORKS
    vae1 = VAE(nc=3, ngf=dim_HR, ndf=dim_HR,
               latent_variable_size=latent_size).to(device)
    vae2 = VAE(nc=3, ngf=dim_LR, ndf=dim_LR,
               latent_variable_size=latent_size).to(device)
    net_face = Net_Attr().to(device)
    net_gender = Net_Attr().to(device)
    net_pale = Net_Attr().to(device)
    net_young = Net_Attr().to(device)

    # OPTIMIZERS
    optimizer1 = optim.Adam(vae1.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer2 = optim.Adam(vae2.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizerface = optim.Adam(
        net_face.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizergender = optim.Adam(
        net_gender.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizerpale = optim.Adam(
        net_pale.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizeryoung = optim.Adam(
        net_young.parameters(), lr=learning_rate, weight_decay=0.0005)

    checkpoint = {'epoch': 0}
'''
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar, img_size):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, img_size), reduction='sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return BCE + KLD
'''

## VAE RECONSTRUCTION LOSS ##

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # # https://arxiv.org/abs/1312.6114 (Appendix B)
    # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar) # original
    # KLD = torch.sum(KLD_element).mul_(-0.5) # original
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)) # modified
    return (BCE + KLD)
    # return (BCE + KLD)/10000 # 10,000 has been added; must verify????????

## DELETE LATER

for i, data in enumerate(train_loader):
    input1, input2, label, gender, pale, young = data.items()
    input1, input2 = input1[1].to(device), input2[1].to(device)
    label1 = label[1].to(device)

    res1, mu1, logvar1, z1 = vae1(input1)
    display_img(input1,dim_HR)
    display_img(res1,dim_HR)
    break

## TRAINING NETWORK ##

if (test_net==False):

    print('===================TRAINING=====================')

    err_rcnstrctn1 = []
    err_rcnstrctn2 = []
    err_same = []  # error of face matching
    err_gender = []  # error of gender matching
    err_pale = []  # error of skin matching
    err_young = []  # error of age matching

    # accuracy of features
    acc_face = []
    acc_gender = []
    acc_pale = []
    acc_young = []


    # minimization criterion of face matching NN
    criterionface = nn.CrossEntropyLoss()
    # maximization criterion of matching NN
    criteriongender = nn.CrossEntropyLoss()
    criterionpale = nn.CrossEntropyLoss()
    criterionyoung = nn.CrossEntropyLoss()

    
    # run for X epochs
    for ep in range(checkpoint['epoch'], epochs):
        # run for every 
        for i, data in enumerate(train_loader):
            # ====================labels=====================

            # input1 = high res image
            # input2 = low res 50% same image
            # label = name corresponding to true image path
            # gender, where male == 1
            input1, input2, label, gender, pale, young = data.items()
            input1, input2 = input1[1].to(device), input2[1].to(device)
            label1 = label[1].to(device)
            gender = gender[1].to(device)
            gender = (gender+1)//2
            pale = pale[1].to(device)
            pale = (pale+1)//2
            young = young[1].to(device)
            young = (young+1)//2
            
            # ===================forward=====================

            vae1.train()
            vae2.train()

            res1, mu1, logvar1, z1 = vae1(input1)
            res2, mu2, logvar2, z2 = vae2(input2)

            z = z1-z2 # vector arithmetic

            # =====================loss=======================

            # feature loss
            lossface = criterionface(net_face(z), label1)
            lossgender = criteriongender(net_gender(z), gender)
            losspale = criterionpale(net_pale(z), pale)
            lossyoung = criterionyoung(net_young(z), young)
            
            # Lface - ΣLs
            lossfeatures = lossface - (lossgender.mul(1) + lossyoung.mul(1) + losspale.mul(1))

            # vae loss calculations = Lreconstruction + Lface - ΣLs
            # parameters: x_hat, x, mu, logvar
            reconstruct1loss = loss_function(res1, input1, mu1, logvar1)
            reconstruct2loss = loss_function(res2, input2, mu2, logvar2)
            encoder1loss = reconstruct1loss + lossfeatures
            encoder2loss = reconstruct2loss + lossfeatures

            # ===================backward====================

            # Zeros descent gradient to help generalize after each batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizerface.zero_grad()
            optimizergender.zero_grad()
            optimizerpale.zero_grad()
            optimizeryoung.zero_grad()

            # feature-based
            lossface.backward(retain_graph=True)
            optimizerface.step()

            lossgender.backward(retain_graph=True)
            optimizergender.step()

            losspale.backward(retain_graph=True)
            optimizerpale.step()

            lossyoung.backward(retain_graph=True)
            optimizeryoung.step()

            # encoder1loss.backward(retain_graph=True)
            reconstruct1loss.backward(retain_graph=True)
            optimizer1.step()
            
            # encoder2loss.backward(retain_graph=False)
            reconstruct2loss.backward(retain_graph=False)
            optimizer2.step()

            # ===================log========================

            # adding loss as an entry to face matching list
            err_rcnstrctn1.append(reconstruct1loss.item())
            err_rcnstrctn1.append(reconstruct2loss.item())
            err_same.append(lossface.item())
            err_gender.append(lossgender.item())
            err_pale.append(losspale.item())
            err_young.append(lossyoung.item())

            # if 5th iteration, print the loss, iteration #, and epoch for each of the NN
            if (i % 5 == 0):
                print(i, ep)
                print(reconstruct1loss,reconstruct2loss)
                print(lossface, lossgender, losspale, lossyoung)
                print(encoder1loss,encoder2loss)
                #print (output[1])

            # if 50th iteration, check accuracy of model
            if (i % 25 == 0):
                # evaluate models; reduces runtime
                vae1.eval()
                vae2.eval()
                # create a numpy array from the indexes from the validation array
                val_index1 = np.random.permutation(val_index)[:100]
                # test for accuracy using 30 images from validation data
                val_dataloader = DataLoader(
                    dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_index1))
                # make validation data iterable
                val_iter = iter(val_dataloader)

                total = 0
                correctface = 0
                correctgender = 0
                correctpale = 0
                correctyoung = 0
                # literally just going through and checking if the model predicts the true value (same face / which gender) correctly
                for j, dataj in enumerate(val_dataloader):
                    input1j, input2j, labelj, gender, pale, young = dataj.items()
                    input1j, input2j = input1j[1].to(
                        device), input2j[1].to(device)

                    labelj = labelj[1].to(device)
                    gender = gender[1].to(device)
                    gender = (gender+1)//2
                    pale = pale[1].to(device)
                    pale = (pale+1)//2
                    young = young[1].to(device)
                    young = (young+1)//2

                    res1, mu1, logvar1, z1 = vae1(input1j)
                    res2, mu2, logvar2, z2 = vae2(input2j)

                    # DISPLAY IMAGES (COMMENT OUT LATER!!!!)
                    display_img(input1j,dim_HR)
                    display_img(res1,dim_HR)

                    _, predictedface = torch.max(net_face(z).data, 1)
                    _, predictedgender = torch.max(net_gender(z).data, 1)
                    _, predictedpale = torch.max(net_pale(z).data, 1)
                    _, predictedyoung = torch.max(net_young(z).data, 1)

                    total += labelj.size(0)
                    correctface += (predictedface == labelj).sum().item()
                    correctgender += (predictedgender == gender).sum().item()
                    correctpale += (predictedpale == pale).sum().item()
                    correctyoung += (predictedyoung == young).sum().item()
                # prints out percent accurate for both NN
                print('Same Face Accuracy: %d %%' % (100*correctface/total))
                print('Gender Male Accuracy: %d %%' % (100*correctgender/total))
                print('Pale Skin Accuracy: %d %%' % (100*correctpale/total))
                print('Young Age Accuracy: %d %%' % (100*correctyoung/total))

                # record the percent accurate for both NN into respective lists
                acc_face.append(100*correctface/total)  # low high
                acc_gender.append(100*correctgender/total)  # male female
                acc_pale.append(100*correctpale/total)  
                acc_young.append(100*correctyoung/total)  

            # DISPLAY IMAGES (COMMENT OUT LATER!!!!)
            if (i % 50 == 0):
                for j, dataj in enumerate(val_dataloader):
                    input1j, input2j, labelj, gender, pale, young = dataj.items()
                    input1j = input1j[1].to(device)
                    labelj = labelj[1].to(device)

                    res1, mu1, logvar1, z1 = vae1(input1j)
                    
                    display_img(input1j,dim_HR)
                    display_img(res1,dim_HR)

        # outside of inner loop
        # saves for every epoch

        # save the checkpoint to be loaded for next use
        save_checkpoint({
            # increase the track of the current epoch #
            'epoch': ep + 1,
            # save Python dictionary object that maps each layer to its parameter tensor.
            'state_dict': vae1.state_dict(),
            # update optimizer conditions (because it may not be zeroed)
            'optimizer': optimizer1.state_dict(),
        })
        save_checkpoint_A({
            'epoch': ep + 1,
            'state_dict': net_face.state_dict(),

            'optimizer': net_face.state_dict(),
        })

## TESTING MODEL ##

if test_net:
    print('===================TESTING=====================')
    err_same = []
    err_gender = []

    acc_face = []
    acc_gender = []
    acc_pale = []
    acc_young = []

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterionpale = nn.CrossEntropyLoss()
    criterionyoung = nn.CrossEntropyLoss()
    
    # val_dataloader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(test_index))

    val_index1 = np.random.permutation(val_index)[:100]
    val_dataloader = DataLoader(
        dataset, batch_size=4, sampler=SubsetRandomSampler(test_index))
    val_iter = iter(val_dataloader)
    predicted_lables = []
    real_labels_g = []
    predicted_lables_g = []
    real_labels = []

    total = 0
    correctface = 0
    correctgender = 0
    correctpale = 0
    correctyoung = 0
    for j, dataj in enumerate(val_dataloader):
        input1j, input2j, labelj, gender, pale, young = dataj.items()
        input1j, input2j = input1j[1].to(device), input2j[1].to(device)

        labelj = labelj[1].to(device)
        gender = gender[1].to(device)
        gender = (gender+1)//2
        pale = pale[1].to(device)
        pale = (pale+1)//2
        young = young[1].to(device)
        young = (young+1)//2

        res1, mu1, logvar1, z1 = vae1(input1j)
        res2, mu2, logvar2, z2 = vae2(input2j)

        _, predictedface = torch.max(net_face(z).data, 1)
        _, predictedgender = torch.max(net_gender(z).data, 1)
        _, predictedpale = torch.max(net_pale(z).data, 1)
        _, predictedyoung = torch.max(net_young(z).data, 1)

        total += labelj.size(0)
        correctface += (predictedface == labelj).sum().item()
        correctgender += (predictedgender == gender).sum().item()
        correctpale += (predictedpale == pale).sum().item()
        correctyoung += (predictedyoung == young).sum().item()

        predicted_lables.append(torch.Tensor.numpy(predictedface.cpu()))
        real_labels.append(torch.Tensor.numpy(labelj.cpu()))

        predicted_lables_g.append(torch.Tensor.numpy(predictedgender.cpu()))
        real_labels_g.append(torch.Tensor.numpy(gender.cpu()))
        print('Same Face Accuracy: %d %%' % (100*correctface/total))
        print('Gender Male Accuracy: %d %%' % (100*correctgender/total))
        print('Pale Skin Accuracy: %d %%' % (100*correctpale/total))
        print('Young Age Accuracy: %d %%' % (100*correctyoung/total))
        acc_face.append(100*correctface/total)
        acc_gender.append(100*correctgender/total)
        print(mt.confusion_matrix(np.array(real_labels).flatten(),
                                  np.array(predicted_lables).flatten()))
        print(mt.confusion_matrix(np.array(real_labels_g).flatten(),
                                  np.array(predicted_lables_g).flatten()))
