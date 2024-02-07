import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# # data
# x_anchor = [1, 2, 3, 4, 5, 6]
# # y_similarity_deg0 = [1.0000, 0.5453, 0.5813, 0.5442, 0.5805, 0.6067]
# # y_similarity_deg10 = [0.9990, 0.5384, 0.5769, 0.5371, 0.5762, 0.6044]
# # y_similarity_deg45 = [0.9917, 0.5337, 0.5710, 0.5320, 0.5706, 0.5985]
# # y_similarity_deg90 = [0.9954, 0.5479, 0.5766, 0.5462, 0.5797, 0.6068]
# y_similarity_deg0 = [0.5805, 0.5098, 0.6637, 0.5060, 1.0000, 0.5847]
# y_similarity_deg10 = [0.5762, 0.5267, 0.6556, 0.5584, 0.9864, 0.5816]
# y_similarity_deg45 = [0.5430, 0.6520, 0.5471, 0.8462, 0.7197, 0.5547]
# y_similarity_deg80 = [0.5314, 0.6848, 0.4880, 0.9908, 0.5151, 0.5414]
# y_similarity_deg90 = [0.5441, 0.6803, 0.5099, 1.0000, 0.5059, 0.5542]


# # plotting
# plt.figure(figsize=(10,4))
# plt.plot(x_anchor, y_similarity_deg0, linestyle='solid', label='0 deg')
# plt.plot(x_anchor, y_similarity_deg10, linestyle='dotted', label='10 deg')
# plt.plot(x_anchor, y_similarity_deg45, linestyle='dashdot', label='45 deg')
# plt.plot(x_anchor, y_similarity_deg80, linestyle='dotted', label='80 deg')
# plt.plot(x_anchor, y_similarity_deg90, linestyle='solid', label='90 deg')
# plt.xlabel('Anchor Index')
# plt.ylabel('Feature Similarity')
# plt.legend()
# plt.show()

x_degree = np.arange(0, 365, 10)

y_raw = np.array([[0.5453, 1.0000, 0.5023, 0.6803, 0.5098, 0.5493],
         [0.5661, 0.9871, 0.5143, 0.6689, 0.5543, 0.5714],
         [0.5942, 0.9427, 0.5456, 0.6451, 0.6245, 0.6006],
         [0.6188, 0.8650, 0.5859, 0.6069, 0.7110, 0.6262],
         [0.6296, 0.7689, 0.6233, 0.5603, 0.7987, 0.6377],
         [0.6258, 0.6741, 0.6483, 0.5163, 0.8730, 0.6337],
         [0.6135, 0.5948, 0.6609, 0.4833, 0.9279, 0.6202],
         [0.6001, 0.5428, 0.6655, 0.4709, 0.9653, 0.6050],
         [0.5882, 0.5147, 0.6662, 0.4776, 0.9898, 0.5924],
         [0.5805, 0.5097, 0.6638, 0.5059, 1.0000, 0.5847],
         [0.5762, 0.5266, 0.6557, 0.5583, 0.9864, 0.5815],
         [0.5696, 0.5576, 0.6375, 0.6309, 0.9419, 0.5771],
         [0.5611, 0.5994, 0.6066, 0.7209, 0.8645, 0.5707],
         [0.5493, 0.6374, 0.5666, 0.8085, 0.7675, 0.5606],
         [0.5374, 0.6636, 0.5286, 0.8795, 0.6747, 0.5491],
         [0.5294, 0.6784, 0.4987, 0.9321, 0.5982, 0.5404],
         [0.5268, 0.6846, 0.4844, 0.9681, 0.5451, 0.5371],
         [0.5312, 0.6847, 0.4878, 0.9908, 0.5148, 0.5412],
         [0.5440, 0.6803, 0.5097, 1.0000, 0.5057, 0.5541],
         [0.5648, 0.6689, 0.5537, 0.9871, 0.5169, 0.5752],
         [0.5927, 0.6451, 0.6236, 0.9424, 0.5473, 0.6035],
         [0.6174, 0.6065, 0.7111, 0.8639, 0.5879, 0.6285],
         [0.6300, 0.5602, 0.7987, 0.7676, 0.6253, 0.6399],
         [0.6278, 0.5157, 0.8729, 0.6729, 0.6502, 0.6362],
         [0.6163, 0.4817, 0.9274, 0.5943, 0.6620, 0.6237],
         [0.6022, 0.4678, 0.9649, 0.5426, 0.6665, 0.6099],
         [0.5898, 0.4740, 0.9897, 0.5148, 0.6668, 0.5981],
         [0.5816, 0.5025, 1.0000, 0.5102, 0.6639, 0.5909],
         [0.5768, 0.5554, 0.9861, 0.5270, 0.6552, 0.5872],
         [0.5701, 0.6287, 0.9410, 0.5578, 0.6363, 0.5807],
         [0.5610, 0.7198, 0.8625, 0.5998, 0.6048, 0.5716],
         [0.5489, 0.8084, 0.7637, 0.6376, 0.5641, 0.5585],
         [0.5369, 0.8799, 0.6692, 0.6634, 0.5255, 0.5445],
         [0.5288, 0.9324, 0.5925, 0.6778, 0.4962, 0.5345],
         [0.5269, 0.9682, 0.5402, 0.6842, 0.4829, 0.5311],
         [0.5321, 0.9908, 0.5110, 0.6846, 0.4873, 0.5357],
         [0.5453, 1.0000, 0.5023, 0.6803, 0.5098, 0.5493],
        ])
y_raw = y_raw.T
print('y_raw', y_raw.shape)
y_A1 = np.array(y_raw[0, :])
y_A2 = np.array(y_raw[1, :])
y_A3 = np.array(y_raw[2, :])
y_A4 = np.array(y_raw[3, :])
y_A5 = np.array(y_raw[4, :])
y_A6 = np.array(y_raw[5, :])

# plotting
plt.figure(figsize=(10,3))
# plt.plot(x_degree, y_A1, linestyle='dotted', label='Anchor 1')
plt.plot(x_degree, y_A2, linestyle='solid', color = '#FFD700', linewidth=3, label='Anchor 1')
plt.plot(x_degree, y_A3, linestyle='solid', color = 'g', linewidth=3, label='Anchor 2')
plt.plot(x_degree, y_A4, linestyle='solid', color = '0.3', linewidth=3, label='Anchor 3')
plt.plot(x_degree, y_A5, linestyle='solid', color = 'r', linewidth=3, label='Anchor 4')
# plt.plot(x_degree, y_A6, linestyle='dotted', label='Anchor 6')
plt.xlabel('Rotation Angle (degree)')
plt.ylabel('Feature Similarity')
plt.legend(loc='right')
plt.show()