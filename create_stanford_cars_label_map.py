import sys
import scipy.io as sio

if (len(sys.argv) < 2):
  print('Missing argument: path to cars_annos.mat file')
  sys.exit(1)

annotation_file = sys.argv[1]
mat = sio.loadmat(annotation_file)

with open('stanford_cars_label_map.pbtxt','w') as output:
  for i, vehicle_class in enumerate(mat['class_names'][0]):
    print(i+1, str(vehicle_class[0]))
    output.write('item {{\n  id: {}\n  name: \'{}\'\n}}\n\n'.format(i+1, vehicle_class[0]))
