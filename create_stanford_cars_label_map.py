import sys
import scipy.io as sio

# TODO input path to mat file
mat = sio.loadmat('/data/StanfordCars/cars_annos.mat')

with open('stanford_cars_label_map.pbtxt','w') as output:
  for i, vehicle_class in enumerate(mat['class_names'][0]):
    print(i+1, str(vehicle_class[0]))
    output.write('item {{\n  id: {}\n  name: \'{}\'\n}}\n\n'.format(i+1, vehicle_class[0]))

  

