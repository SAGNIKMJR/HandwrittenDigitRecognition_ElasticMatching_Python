import os
import cv2
import numpy as np
from PIL import Image
from mnist import MNIST

def deskew(source_dir, save_dir=None, negate=False, second_moment_threshold=1e-2):
    """
    This method deskwes images in a directory using moments and store them back
    :param source_dir: directory where images are contained with no sub-directories
    :param save_dir: directory where resized images are to be stored (default: None (stores in the source))
    :param negate: a boolean flag telling  whether the input image is to be negated
    :param second_moment_threshold: a float threshold which decides if the image is to be deskewed
    """

    # saving in source_dir if source_dir param is None
    if save_dir is None:
        save_dir = source_dir

    # directory traversal
    file_list=[]
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        file_list.extend(filenames)

    # deskewing and storing back
    for image_file in file_list:
        image = Image.open(os.path.join(source_dir, image_file))

        # negate the image
        if negate:
           image = 255-image

        # image->numpy array and calculation of moments
        image=np.array(image)
        moments = cv2.moments(image)

        # if second order moments < threshold, don't skew
        if abs(moments['mu02']) < second_moment_threshold:
            image.save(os.path.join(save_dir, image_file), 'png')
            return 

        # computing skew and deskewing
        skew = moments['mu11']/moments['mu02']
        mask = numpy.float32([[1, skew, -0.5*image.shape[0]*skew], [0,1,0]])
        image = cv2.warpAffine(image, mask, image.shape, flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        image.save(os.path.join(save_dir, image_file), 'png')

def resize(source_dir, resized_size=20, save_dir=None):
    """
    This method resizes images using PIL.Image
    :param source_dir: directory where images are contained with no sub-directories
    :param size: an integer denoting the size to resize to (default)
    :param save_dir: directory where resized images are to be stored (default: None (stores in the source))
    """

    # saving in source_dir if source_dir param is None
    if save_dir is None:
        save_dir = source_dir

    # directory traversal
    file_list=[]
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        file_list.extend(filenames)

    # resizing and storing back
    for image_file in file_list:
        image = Image.open(os.path.join(source_dir, image_file))
        image = image.resize(resized_size, Image.ANTIALIAS)
        image.save(os.path.join(save_dir, image_file), 'png')

def binarize(source_dir, save_dir=None, binarization_threshold=0):
    """
    DISCLAIMER: don't use binarization on input images, doesn't help
    This method deskwes images in a directory using moments and store them back
    :param source_dir: directory where images are contained with no sub-directories
    :param save_dir: directory where resized images are to be stored (default: None (stores in the source))
    :param binarization_threshold: a float threshold for binarization
    """

    # saving in source_dir if source_dir param is None
    if save_dir is None:
        save_dir = source_dir

    # directory traversal
    file_list=[]
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        file_list.extend(filenames)

    # binarizing and storing back
    for image_file in file_list:
        image = Image.open(os.path.join(source_dir, image_file))

        # image->numpy array and binarization
        image=np.array(image)
        image = image>binarization_threshold
        image.save(os.path.join(save_dir, image_file), 'png')

def extract(dataset = 'MNIST', raw_data = None):
    """
    This method extracts images using the Python MNIST dataparser (https://github.com/sorki/python-mnist)
    and stores it in './datasets/MNIST/match' and './datasets/MNIST/eval' respectively 
    :param dataset: name of dataset
    :param raw_data: repository cloned from https://github.com/sorki/python-mnist
    """

    # raise exception if dataset is not MNIST
    if dataset != 'MNIST':
        raise NotImplementedError('datasets other than MNIST not implemented!')

    # extract only if path given to args.raw_data, otherwise use data from
    # './datasets/MNIST/match' and './datasets/MNIST/eval' directly
    if raw_data is not None:
        if ((not os.path.exists('./datasets/MNIST/mask')) and (not os.path.exists('./datasets/MNIST/eval'))):
            os.makedirs('./datasets/MNIST/mask')
            os.makedirs('./datasets/MNIST/eval')
        elif not os.path.exists('./datasets/MNIST/mask'):
            os.makedirs('./datasets/MNIST/mask')
        elif not os.path.exists('./datasets/MNIST/eval'):           
            os.makedirs('./datasets/MNIST/eval')
        mask_dir = './datasets/MNIST/mask'
        eval_dir = './datasets/MNIST/eval'

        mnist_data = MNIST(raw_data)
        mask_images, mask_labels = mnist_data.load_training()
        eval_images, eval_labels = mnist_data.load_testing()

        image_count = 0
        label_count = [1]*10
        for label in mask_labels:
            image = Image.new('L', (28,28), 'white')
            image.put_data(match_images[image_count])
            image_file = str(label) + '_' + str(label_count[label])
            image.save(os.path.join(mask_dir, image_file), 'png') 
            image_count += 1
            label_count[label] += 1

        image_count = 0
        label_count = [1]*10
        for label in eval_labels:
            image = Image.new('L', (28,28), 'white')
            image.put_data(eval_images[image_count])
            image_file = str(label) + str(label_count[label])
            image.save(os.path.join(eval_dir, image_file), 'png') 
            image_count += 1
            label_count[label] += 1