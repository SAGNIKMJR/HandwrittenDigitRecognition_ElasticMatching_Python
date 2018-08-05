import argparse
import math 

parser = argparse.ArgumentParser(description='Handwritten Digit Recognition by Elastic Matching')

"""
     NOTE:
     put data into two directories for matching and evaluation and pass the paths
     to args.match_data and args.eval_data respectively
     DEPENDENCIES:
     i) Python MNIST Parser: https://github.com/sorki/python-mnist
     ii) bob 5.0.0 : https://www.idiap.ch/software/bob/
     iii) OpenCV 3.1.0: https://github.com/opencv/opencv/releases/tag/3.1.0
"""
# Dataset and loading
parser.add_argument('-raw', '--raw-data', default = None,  metavar='RAWDIR',
                    help='path to raw dataset (python-mnist from https://github.com/sorki/python-mnist)')
parser.add_argument('-mask', '--mask-data', default = './datasets/MNIST/mask',  metavar='MASKDIR',
                    help='path to dataset for masking with Gabor filter')
parser.add_argument('-eval', '--eval-data', default = './datasets/MNIST/eval', metavar='EVALDIR',
                    help='path to eval dataset')
parser.add_argument('--dataset', default='MNIST',
                    help='name of dataset (default: MNIST, note: has been tested on MNIST comprehesively)')
parser.add_argument('-r', '--resized-size', default=20, type=int,
                    metavar='R', help='resized size for crops (default: 20)')

# Matching hyper-parameters
parser.add_argument('-ns', '--no-scales', default=2, type=int, metavar='NS',
                    help='number of scales of Gabor filter')
parser.add_argument('-nd', '--no-directions', default=4, type=int, metavar='ND',
                    help='number of directions of Gabor filter')
parser.add_argument('-s', '--sigma', default=2*math.pi, type=float,
                    metavar='S', help='width/spatial resolution of \
                    Gaussian envelope of Gabor wavelet (default: 2*Pi)')
parser.add_argument('-fm', '--frequency-max', default=math.pi, type=float, metavar='FM',
                    help='highest frequency level of Gabor wavelet(default: Pi)')
parser.add_argument('-ff', '--frequency-factor', default=0.5, type=float,
                    metavar='FF', help='square root ratio of two scales of Gabor wavelets (default: 0.5)')
parser.add_argument('-rw', '--relative-weight', default=3e-9, type=float,
                    metavar='RW', help='relative weight between two loss terms (default:3e-9)')
parser.add_argument('-mi', '--match-ite', default=100, type=int,
                    metavar='MI', help='iterations for matching with a single image mask (default: 100)')
parser.add_argument('-pf', '--print-freq', default=50, type=int,
                    metavar='PF', help='print frequency (default: 50)')
