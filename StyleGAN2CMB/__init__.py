"""
"""

""" Setup gpu details """
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpus = ['0','1']
if len(gpus) > 0:
    gpus = np.hstack(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

""" Local refs """
from .dataset import CMBDataset
from .stylegan2 import StyleGAN, AdvTranslationNet




