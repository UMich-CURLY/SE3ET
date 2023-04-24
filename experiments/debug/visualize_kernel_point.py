import numpy as np
from geotransformer.modules.e2pn.ply import read_ply

def normalize(pc, radius):
    r = np.sqrt((pc**2).sum(1).max())
    return pc*radius/r

# k_015_center_3D, k_015_verticals_3D
# FILE_NAME = '/home/cel/code/GeoTransformer/geotransformer/modules/e2pn/vgtk/vgtk/data/anchors/k_015_center_3D.ply'
# OUTPUT_FILE_NAME = '/home/cel/code/GeoTransformer/geotransformer/modules/e2pn/vgtk/vgtk/data/anchors/k_015_center_3D.bin'
# data = read_ply(FILE_NAME)
# points = np.vstack((data['x'], data['y'], data['z'])).T
# print('points', points.shape, '\n', points)

# output_file = open(OUTPUT_FILE_NAME, "bw")
# points.tofile(output_file)


# kpsphere24, kpsphere30, kpsphere66
FILE_NAME = '/home/cel/code/GeoTransformer/geotransformer/modules/e2pn/vgtk/vgtk/data/anchors/kpsphere24.ply'
OUTPUT_FILE_NAME = '/home/cel/code/GeoTransformer/geotransformer/modules/e2pn/vgtk/vgtk/data/anchors/kpsphere24.bin'

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../geotransformer/modules/e2pn/vgtk') )
import vgtk
import vgtk.pc as pctk
data = pctk.load_ply(FILE_NAME).astype('float32')
# radius = 0.7 * 0.0625
# data = normalize(data, radius)
print('data', data.shape, data.dtype, '\n', data)
data = np.array(data, dtype=np.float64())
print('data', data.shape, data.dtype, '\n', data)
output_file = open(OUTPUT_FILE_NAME, "bw")
data.tofile(output_file)