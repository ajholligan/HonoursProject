import os
import pdb
import time
import random
import shutil
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from torch.utils import data
from torchvision import transforms
from sklearn.model_selection import train_test_split


import PIL
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, sigma, alpha):
    '''
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].
    
    image: a three-dimensional numpy array representing the PIL image
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields
    
    returns: an elastically distorted image of the same shape
    '''
    assert len(image.shape) == 3
    # the two lines below ensure we do not alter the array images
    e_image = np.empty_like(image)
    e_image[:] = image
    height = image.shape[0]
    width = image.shape[1]
    
    random_state = np.random.RandomState(None)
    x, y = np.mgrid[0:height, 0:width]
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
    indices = x + dx, y + dy
    
    for i in range(e_image.shape[2]):
        e_image[:, :, i] = map_coordinates(e_image[:, :, i], indices, order=1)

    return e_image

class MNISTElasticTranform(object):
    """
    Applies Elastic Transform on MNIST images
    """

    def __init__(self, sigma, alpha):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, sample):
        image = sample
        image = np.asarray(image).reshape(28, 28, 1)
        image = elastic_transform(image, self.sigma, self.alpha)
        image = PIL.Image.fromarray(image.reshape(28, 28))

        return image

class CIFAR10ElasticTransform(object):
    """
    Applies Elastic Transform on CIFAR images
    """

    def __init__(self, sigma, alpha):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, sample):
        image = sample
        image = np.asarray(image)
        image = elastic_transform(image, self.sigma, self.alpha)
        image = PIL.Image.fromarray(image)

        return image

def get_mnist_loaders(data_dir, bsize, num_workers, sigma, alpha):
    transform = transforms.Compose([MNISTElasticTranform(sigma, alpha), transforms.ToTensor()])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True,
        transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=bsize, num_workers=num_workers)

    valid_set = datasets.MNIST(root=data_dir, train=False, download=True,
        transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=bsize, num_workers=num_workers)

    return train_loader, valid_loader

def get_cifar_loaders(data_dir, bsize, num_workers, sigma, alpha):
    transform = transforms.Compose([CIFAR10ElasticTransform(sigma, alpha), transforms.ToTensor()])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=bsize, shuffle=True, num_workers=num_workers)

    valid_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=bsize, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader

def get_fmnist_loaders(data_dir, bsize, num_workers, sigma, alpha):
    transform = transforms.Compose([MNISTElasticTranform(sigma, alpha), transforms.ToTensor()])

    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True,
        transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=bsize, num_workers=num_workers)

    valid_set = datasets.FashionMNIST(root=data_dir, train=False, download=True,
        transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=bsize, num_workers=num_workers)

    return train_loader, valid_loader



class Dataset(data.Dataset):
    """
    Class for the handling of training, validation and testing datasets.
    This class handles dynamically loading and augmenting the datasets.
    """

    def __init__(self, mode, filenames, labels):
        """
        Initiliser for the class that stores the filenames and labels for the model.
        :param arguments: ArgumentParser containing arguments.
        :param mode: String specifying the type of data loaded, "train", "validation" and "test".
        :param filenames: Array of filenames.
        :param labels: Array of labels.
        """

        # Calls the PyTorch Dataset Initiliser.
        super(Dataset, self).__init__()

        # Stores the arguments and mode in the object.
       # self.arguments = arguments
        self.mode = mode

        # Sets the Pillow library to load truncated images.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # Stores the filenames and labels in the object.
        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        """
        Gets the length of the dataset.
        :return: Integer for the length of the dataset.
        """

        return len(self.filenames)

    def __getitem__(self, index):
        """
        Gets a given image and label from the dataset based on a given index.
        :param index: Integer representing the index of the data from the dataset.
        :return: A Tensor containing the augmented image and a integer containing the corresponding label.
        """

        # Loads and augments the image.
        image = Image.open(self.filenames[index])
        image = self.augment(image)

        # Returns the image and label.
        return image, self.labels[index]

    def augment(self, image):
         """
         Method for augmenting a given input image into a tensor.
         :param image: A Pillow Image.
         :return: A augmented image Tensor.
         """

         # Mean and Standard Deviation for normalising the dataset.
         mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

         # Declares the list of standard transforms for the input image.
         augmentations = [transforms.Resize((600, 450), Image.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=mean, std=std)]

         # Adds additional transformations if selected.
        # if self.arguments.augmentation and self.mode == "train":

             # Class for Random 90 degree rotations.
         #    class RandomRotation:
          #       def __init__(self, angles): self.angles = angles
           #      def __call__(self, x):
            #         return transforms.functional.rotate(x, float(np.random.choice(self.angles)))

             # Adds the additional augmentations to the list of augmentations.
             #augmentations = augmentations[:1] + [transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
              #                RandomRotation([0, 90, 180, 270])] + augmentations[1:]

         # Applies the augmentations to the image.
         return transforms.Compose(augmentations)(image)



def get_isic_loaders(data_dir, bsize, num_workers, sigma, alpha):
    """
    Get the training, validation and testing Dataset objects.
    :param arguments: ArgumentParser Namespace object.
    :return: Dataset objects for training, validation and testing.
    """

     # Reads the training data csb file containing filenames and labels.
    csv_file = pd.read_csv(os.path.join("C:\\Users\\aholl\\OneDrive\\Documents\\Honours Project\\Code\\pytorch-prototypeDL-master\\data\\crop", "HAM10000_metadata.csv"))

    # Gets the filenames and labels fom the csv file.
    filenames = csv_file["image_id"].tolist()
    labelsList =   csv_file["dx"].tolist()#np.argmax(np.array(csv_file.drop(["lesion_id", "dx"], axis=1)), axis=1)
    #print(labelsList[0])
    i = 0
    for x in labelsList:
        if(x == 'bkl'):
           labelsList[i] = 0
        elif (x == 'df'):
            labelsList[i] = 1
        elif (x == 'mel'):
            labelsList[i] = 2
        elif (x == 'nv'):
            labelsList[i] = 3
        elif (x == 'vasc'):
            labelsList[i] = 4
        elif (x == 'bcc'):
            labelsList[i] = 5
        elif (x == 'akiec'):
            labelsList[i] = 6
        i +=1

        
    #print(labelsList)
    labels = torch.tensor(labelsList)
    #Tensor(labelsList)
    # Adds the file path to each filename.
    for i in range(len(filenames)):
        #print(filenames[i])
        filenames[i] = f"C:\\Users\\aholl\\OneDrive\\Documents\\Honours Project\\Code\\pytorch-prototypeDL-master\\data\\crop\\{filenames[i]}.jpg"
        
    # Splits the dataset into training and testing.
    val_filenames, filenames, val_labels, labels = train_test_split(filenames, labels,
                                                                        test_size=0.2 + 0.7,
                                                                        random_state= 3)

    # Splits the testing dataset into testing and validation.
   # test_filenames, val_filenames, test_labels, val_labels = train_test_split(filenames, labels,
    #                                                                          test_size=arguments.validation_split,
     #                                                                         random_state=arguments.seed)

    # Creates the training, validation and testing dataset objects.
    train_data = Dataset("train", filenames, labels)
    val_data = Dataset("validation", val_filenames, val_labels)
    #print(val_data)
    #test_data = Dataset(arguments, "test", test_filenames, test_labels)
    valid_loader = DataLoader(val_data, shuffle=False, batch_size=bsize, num_workers=num_workers)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=bsize, num_workers=num_workers)
    # Returns the dataset objects.
    return train_loader, valid_loader#, test_data
