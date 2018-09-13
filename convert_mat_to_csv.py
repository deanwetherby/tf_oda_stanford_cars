import sys
import os
import csv
import numpy as np
import scipy.io as sio

mat_file = sys.argv[1]
csv_file = sys.argv[2]

with open(csv_file, 'wb') as csvfile:

  mat = sio.loadmat(mat_file)

  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(['relative_im_path','class','bbox_x1','bbox_y1','bbox_x2','bbox_y2','test'])

  for annotation in mat['annotations'][0]:
    test = np.squeeze(annotation['test'])
    im_path = str(np.squeeze(annotation['relative_im_path']))
    cls = np.squeeze(annotation['class'])
    x1 = np.squeeze(annotation['bbox_x1'])
    y1 = np.squeeze(annotation['bbox_y1'])
    x2 = np.squeeze(annotation['bbox_x2'])
    y2 = np.squeeze(annotation['bbox_y2'])

    csvwriter.writerow([im_path, cls, x1, y1, x2, y2, test])

