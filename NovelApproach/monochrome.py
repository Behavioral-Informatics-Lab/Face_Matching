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
dim_LR = 128
shape = (dim_LR, dim_LR)
dim_HR = 128
ip_shape = (dim_HR, dim_HR)
batch_size = 50
learning_rate = 1e-3
epochs = 5
latent_size = 500
# d = 32
train_ratio = .7
val_ratio = .3
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
            skin = col['Pale_Skin'].values[0]
            age = col['Young'].values[0]

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
            skin = col['Pale_Skin'].values[0]
            age = col['Young'].values[0]

        # annot = annot.astype('float').reshape(-1,2)
        sample = {'image1': image1, 'image2': image2,
                  'same': annot, 'gender': gender,
                  'skin': skin, 'age': age}
        return sample


columns = ['ImgId', '5_o_Clock_Shadow', ' Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
           'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
cele_attrib = pd.read_csv(file_path, delimiter="\s+", names=columns)

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

# CHECKPOINTS #

def save_checkpoint(state, filename='Checkpoints/checkpoint_0.001.pth.tar'):
    torch.save(state, filename)
# def save_checkpoint_A(state, filename='/home/jamal/Downloads/additional_files/checkpoint_A_0.001.pth.tar'):

def save_checkpoint_A(state, filename='Checkpoints/checkpoint_A_0.001.pth.tar'):
    torch.save(state, filename)

# NEURAL NETWORKS #


class Net(nn.Module):
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

        # change the first parameter in case you change the size of the small image
        self.ip1 = nn.Linear(d, 8)
        self.ip2 = nn.Linear(8, 2)

        self.encoder2 = nn.Sequential(
            nn.Linear(12*12, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 12*12),  # rconstructed image
            nn.Sigmoid(),
        )

        '''
        self.ip3 = nn.Linear(300*20,20,False)   # change the first parameter in case you change the size of the small image
        self.ip4 = nn.Linear(20,2,False)
        '''

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x1, x2):
        # HR VAE
        # print(x1.shape) # three color channels for R,G,B.

        encoded1 = self.encoder1(x1.view(-1, 150*150))
        mu_logvar1 = encoded1.view(-1, 2, d)
        print(mu_logvar1.shape)
        mu1 = mu_logvar1[:, 0, :]
        logvar1 = mu_logvar1[:, 1, :]
        z1 = self.reparameterise(mu1, logvar1)
        recnstrct1 = self.decoder1(z1)

        # LR VAE
        mu_logvar2 = self.encoder2(x2.view(-1, 12*12)).view(-1, 2, d)
        mu2 = mu_logvar2[:, 0, :]
        logvar2 = mu_logvar2[:, 1, :]
        z2 = self.reparameterise(mu2, logvar2)
        recnstrct2 = self.decoder2(z2)

        '''
        in_pic = x1.data.cpu().view(-1, 150, 150)
        plt.figure(figsize=(18, 4))
        plt.imshow(in_pic[0])
        plt.axis('off')
        plt.show()

        in_pic = recnstrct1.data.cpu().view(-1, 150, 150)
        plt.figure(figsize=(18, 4))
        plt.imshow(in_pic[0])
        plt.axis('off')
        plt.show()

        in_pic = x2.data.cpu().view(-1, 12, 12)
        plt.figure(figsize=(18, 4))
        plt.imshow(in_pic[0])
        plt.axis('off')
        plt.show()

        in_pic = recnstrct2.data.cpu().view(-1, 12, 12)
        plt.figure(figsize=(18, 4))
        plt.imshow(in_pic[0])
        plt.axis('off')
        plt.show()
        '''

        # vector arithmetic
        z = z1-z2

        # takes z vector and reduces it to two nodes (same face or not)
        facesame = self.ip1(z)
        facesame = F.relu(facesame)
        facesame = self.ip2(facesame)

        return facesame, z, recnstrct1, mu1, logvar1, recnstrct2, mu2, logvar2


# adversarial neural network (male/female identifier)
class Net_A(nn.Module):

    def __init__(self):
        super(Net_A, self).__init__()

        self.ip3 = nn.Linear(
            # input = size of flattened image
            d,
            # output (heavily condensing)
            8,
            # bias set to False, the layer will not learn an additive bias (Default: True)
            False)   # change the first parameter in case you change the size of the small image

        # Relu between layers

        self.ip4 = nn.Linear(8,
                             # output corresponds to two nodes: one for female and the other for male
                             2,
                             # bias set to False, the layer will not learn an additive bias (Default: True)
                             False)

    def forward(self, z):
        # takes z for predicting male or not
        malefem = self.ip3(z)
        malefem = F.relu(malefem)
        malefem = self.ip4(malefem)
        # x2 = x2.mul(-1)
        # x = F.relu(x)
        # x = F.softmax(x,1)
        return malefem


class Net_Attr(nn.Module):

    def __init__(self):
        super(Net_Attr, self).__init__()

        self.ip1 = nn.Linear(
            # input = size of flattened image
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
        malefem = self.ip1(z)
        malefem = F.relu(malefem)
        malefem = self.ip2(malefem)
        # x2 = x2.mul(-1)
        # x = F.relu(x)
        # x = F.softmax(x,1)
        return malefem

# LOADING / INITIALIZING NEURAL NETWORK #


if load_net:
    net = VAE().to(device)
    net_A = Net_A().to(device)

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
    net_skin = Net_Attr().to(device)
    net_age = Net_Attr().to(device)

    # OPTIMIZER
    optimizer1 = optim.Adam(vae1.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer2 = optim.Adam(vae2.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizerface = optim.Adam(
        net_face.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizergender = optim.Adam(
        net_gender.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizergender = optim.Adam(
        net_skin.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizergender = optim.Adam(
        net_age.parameters(), lr=learning_rate, weight_decay=0.0005)

    checkpoint = {'epoch': 0}

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar, img_size):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, img_size), reduction='sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return BCE + KLD




if (test_net==False):
    # error
    err_rcnstrctn1 = []
    err_rcnstrctn2 = []
    err_same = []  # error of face matching
    err_gender = []  # error of gender matching
    err_skin = []  # error of skin matching
    err_age = []  # error of age matching

    # accuracy of face matching
    acc1 = []
    # accuracy of gender matching
    acc2 = []


    # minimization criterion of face matching NN
    criterionface = nn.CrossEntropyLoss()
    # maximization criterion of gender matching NN
    criteriongender = nn.CrossEntropyLoss()

    # not sure what t is ? it's only use seems to be commented out
    t = -1
    print('start')
    # run for 5 epochs
    for ep in range(checkpoint['epoch'], epochs):
        for i, data in enumerate(train_loader):

            # input1 = high res true image
            # input2 = low res true or false image
            # label = name corresponding to true image path
            # gender = male or female label
            input1, input2, label, gender = data.items()
            # run on cuda gpu if available
            input1, input2 = input1[1].to(device), input2[1].to(device)
            label1 = label[1].to(device)
            gender = gender[1].to(device)
            # not sure why it is (gender+1)/2?
            gender = (gender+1)//2
            #label = torch.Tensor.long(label[1])

            # ===================forward=====================

            # feed input images 1 and 2 into neural network
            output = net(input1,input2)
            # feed output of main face matching NN to gender matching NN

            output_A = net_A(output[1])
            
            # =====================loss=======================

            # parameters: x_hat, x, mu, logvar, shape
            reconstruct1loss = loss_function(res1, input1, mu1, logvar1)
            reconstruct2loss = loss_function(res2, input2, mu2, logvar2)
            lossface = criterionface(net_face(z), label1)
            lossgender = criteriongender(net_gender(z), gender)

            # vae loss calculations
            encoder1loss = reconstruct1loss + lossface - lossgender.mul(1)
            encoder2loss = reconstruct2loss + lossface - lossgender.mul(1)

            # ===================backward====================

            # Zeros descent gradient to help generalize after each batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizerface.zero_grad()
            optimizergender.zero_grad()

            # reconstruction loss 
            # facesame, cat, recnstrct1, mu1, logvar1, recnstrct2, mu2, logvar2
            reconstruct1loss = loss_function(output[2], input1, output[3], output[4], 150*150)
            reconstruct2loss = loss_function(output[5], input2, output[6], output[7], 12*12)

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

            # formula: L = Ly + Lhl + Llr - wLd
            loss1 = loss1 + reconstruct1loss + reconstruct2loss - loss2.mul(1)  #new line 
            #loss2 = loss2.abs()
            # t = t*-1

            # backwards propagation 
            loss1.backward()
            # face matching NN gradient step
            optimizer.step()

            # ===================log========================

            # adding loss as an entry to face matching list
            err_rcnstrctn1.append(reconstruct1loss.item())
            err_rcnstrctn1.append(reconstruct2loss.item())
            err_same.append(lossface.item())
            err_gender.append(lossgender.item())

            # evaluate models; reduces runtime
            vae1.eval()
            vae2.eval()

            # if 5th iteration, print the loss, iteration #, and epoch for each of the NN
            if (i % 5 == 0):
                print(reconstruct1loss, i, ep)
                print(reconstruct2loss, i, ep)
                print(lossface, i, ep)
                print(lossgender, i, ep)
                #print (output[1])

            # if 50th iteration, check accuracy of model
            if (i % 25 == 0):
                # create a numpy array from the indexes from the validation array
                val_index1 = np.random.permutation(val_index)[:100]
                # test for accuracy using 30 images from validation data
                val_dataloader = DataLoader(
                    dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_index1))
                # make validation data iterable
                val_iter = iter(val_dataloader)

                total = 0
                correct1 = 0
                correct2 = 0
                # literally just going through and checking if the model predicts the true value (same face / which gender) correctly
                for j, dataj in enumerate(val_dataloader):
                    input1j, input2j, labelj, gender = dataj.items()
                    input1j, input2j = input1j[1].to(
                        device), input2j[1].to(device)

                    labelj = labelj[1].to(device)
                    gender = gender[1].to(device)
                    gender = (gender+1)//2

                    res1, mu1, logvar1, z1 = vae1(input1j)
                    res2, mu2, logvar2, z2 = vae2(input2j)
                    _, predicted1 = torch.max(net_face(z).data, 1)
                    _, predicted2 = torch.max(net_gender(z).data, 1)
                    total += labelj.size(0)
                    correct1 += (predicted1 == labelj).sum().item()
                    correct2 += (predicted2 == gender).sum().item()
                # prints out percent accurate for both NN
                print('Face Accuracy: %d %%' % (100*correct1/total))
                print('Gender Accuracy: %d %%' % (100*correct2/total))

                # record the percent accurate for both NN into respective lists
                acc1.append(100*correct1/total)  # low high
                acc2.append(100*correct2/total)  # male female

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


# visualization test

imgplot = plt.imshow(lum_img)
plt.colorbar()

## TESTING MODEL ##

if test_net:
    err_same = []
    err_gender = []

    acc1 = []
    acc2 = []

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    t = -1
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
    correct1 = 0
    correct2 = 0
    for j, dataj in enumerate(val_dataloader):
        input1j, input2j, labelj, gender = dataj.items()
        input1j, input2j = input1j[1].to(device), input2j[1].to(device)

        labelj = labelj[1].to(device)
        gender = gender[1].to(device)
        gender = (gender+1)//2

        res1, mu1, logvar1, z1 = vae1(input1j)
        res2, mu2, logvar2, z2 = vae2(input2j)

        _, predicted1 = torch.max(net_face(z).data, 1)
        _, predicted2 = torch.max(net_gender(z).data, 1)

        total += labelj.size(0)
        correct1 += (predicted1 == labelj).sum().item()
        correct2 += (predicted2 == gender).sum().item()
        predicted_lables.append(torch.Tensor.numpy(predicted1.cpu()))
        real_labels.append(torch.Tensor.numpy(labelj.cpu()))

        predicted_lables_g.append(torch.Tensor.numpy(predicted2.cpu()))
        real_labels_g.append(torch.Tensor.numpy(gender.cpu()))
        print('Accuracy_LH: %d %%' % (100*correct1/total))
        print('Accuracy_MF: %d %%' % (100*correct2/total))
        acc1.append(100*correct1/total)
        acc2.append(100*correct2/total)
        print(mt.confusion_matrix(np.array(real_labels).flatten(),
                                  np.array(predicted_lables).flatten()))

        print(mt.confusion_matrix(np.array(real_labels_g).flatten(),
                                  np.array(predicted_lables_g).flatten()))
