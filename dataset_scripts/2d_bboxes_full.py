import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse

if __name__ == "__main__":
    # First, get 2D bboxes from 3D bboxes
    parser = argparse.ArgumentParser(description="Get 2D bboxes from 3D bboxes")
    parser.add_argument('--split', '-s', type=str, default='routes_training_new', help='Dataset name')
    args = parser.parse_args()

    dataset = "/home/nupdm/Datasets/nuPDM/" + args.split

    
