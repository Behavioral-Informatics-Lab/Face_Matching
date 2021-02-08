# Fairness in Face Matching Algorithms
Jamal, Aidan, Chidansh, and Ahmed

## Papers
| # 	| Paper Title                                 	| Features(s) 	| Implementation            	|
|---	|---------------------------------------------	|----------------	|---------------------------	|
| 1 	| [Toward Fairness in Face Matching Algorithms](https://wp.comminfo.rutgers.edu/vsingh/wp-content/uploads/sites/110/2019/09/Workshop_paper_CameraReady.pdf) 	| Gender         	| Adversarial deep learning 	|

## Python Scripts
| # 	| Title                               	| Contributor(s) 	| Description       	|
|---	|-------------------------------------	|----------------	|-------------------	|
| 1 	| main.py                             	| Jamal          	| Training          	|
| 2 	| main.ipynb                          	| Jamal, Aidan   	| Annotated main.py 	|
| 3 	| low_high_celeb_adversal_test_all.py 	| Jamal          	| CelebA Testing    	|
| 4 	| test_main.py                        	| Jamal          	| UMD Faces Testing 	|

## Data Sets
| Name      	| Description                                                  	|
|-----------	|--------------------------------------------------------------	|
| [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)    	| 200K celebrity images with 40+ attributes.                   	|
| [UMD Faces](https://www.umdfaces.io/) 	| University of Maryland images with gender and box feature labels. 	|

Additional data sets can be found on [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121594)

## Related Works
- [Joint Feature and Similarity Deep Learning for Vehicle Re-identification](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8424333)
- [Variational Fair Autoencoder](https://github.com/dendisuhubdy/vfae/blob/master/vfae.py)
- [A Cloud-guided Feature Extraction Approach for Image Retrieval in Mobile Edge Computing](https://ieeexplore.ieee.org/document/8851250)
- [HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition](https://mail.google.com/mail/u/1/#search/jka59%40scarletmail.rutgers.edu/FMfcgxwKjKqxQGstpSCQttXlsGMFDfDx?projector=1&messagePartId=0.1)
- [Deep Feature Consistent Variational Autoencoder](https://houxianxu.github.io/assets/project/dfcvae)

## Coding Resources
- [Pytorch VAE Git Repo](https://github.com/AntixK/PyTorch-VAE)
- [Theanos VFAE Git Repo](https://github.com/NCTUMLlab/Huang-Ching-Wei/tree/master/Model)

## Amarel Resources
- [Bala's Intro to Amarel Video](https://ru-stream.rutgers.edu/media/Intro+to+the+Amarel+Cluster+-+OARC+Workshop/1_2xsi7647)
- [Kaltura Videos](https://rutgers.mediaspace.kaltura.com/channel/OARC-Weekly-Open-Workshop/171647611)
- [OnDemand](https://ondemand.hpc.rutgers.edu/pun/sys/dashboard)
- [OARC](https://oarc.rutgers.edu/resources/amarel/)
- [User Guide](https://sites.google.com/view/cluster-user-guide/amarel)
- Add the following to SLURM tTo use GPUs:
```
#SBATCH --gres=gpus:1
#SBATCH -partition=gpu
```

## Repo Use Instructions
- To clone the repo, enter below snippet into your command line 
```
git clone https://github.com/Behavioral-Informatics-Lab/Face_Matching.git
```
- Remember to pull to avoid version conflicts!
```
git pull
```
- To push your changes to the git repo, enter the following (or a similar variant) in your command line:
```
git add .
git commit -m "Your comment"
git push
```
- If pushes are taking too long, then consider 
```
git config http.postBuffer 524288000
```
## low_high_celeb_adversal_test_all.py Use Instructions
- Locally install the images ONLY from the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) data set. This should be a zip file. The annotations should already be a part of the repo. The file structure:

```
├── Checkpoints
└── Data Sets
    └── CelebA
        ├── annot
        └── img_align_celeba
            └── [INSERT IMAGES HERE]
    └── UMD Faces
```
- first use (0), after checkpoint saved (1)
```
load_net = 0 
```
- CPU (0), GPU (1)
```
use_cuda = 0 
```
- Modify train, validation, and test ratios accordingly
```
train_ratio = 0
val_ratio = 0
test_ratio = 1-train_ratio-val_ratio
```
- Make sure you are using the correct version of pytorch for running Jamal's code
```
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html 
```
- Run code (here is what my command line looks like)
```
PS C:\Users\austr\Desktop\BIL Face Matching Alg Fairness\Face_Matching> & C:/Users/austr/anaconda3/python.exe "c:/Users/austr/Desktop/BIL Face Matching Alg Fairness/Face_Matching/Paper1Scripts/low_high_celeb_adversal_test_all.py"
```
