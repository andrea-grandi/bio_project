"""
This file is for extract features from patch.

We use Detectron2 (trained by us) for instance segmentation 
and extract the masks and after compute nuclei distribution, 
number of nuclei and cells classification.

With this we can pass the features to the bufferMIL
instead of using significant pathes like the original.

"""

import numpy as np
import torch
import cv2