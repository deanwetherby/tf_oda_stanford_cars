# TensorFlow Object Detection API for Stanford Cars

I am trying to use Google's TensorFlow Object Detection API [0] to detect 196 vehicles types in the Stanford Cars dataset [1]. I'm currently having issues getting this to work with multiple classes. Examples on the web [2][3] use only a single class and appear to work well but I can't seem to find any multiclass examples. As a car detector, the Object Detection API works on the cars dataset.

## Getting Started

Download the Stanford Cars dataset from the link below.
```
http://imagenet.stanford.edu/internal/car196/car_ims.tgz
```

Clone this repo.
```
git clone https://github.com/deanwetherby/tf_oda_stanford_cars
cd tf_oda_stanford_cars
```

Create the train and test tfrecords from the Stanford Cars annotations. Making pbtxt label file contains hard-coded paths but the labels file is already provided for you.

```
(tf) $ python create_stanford_cars_tf_record.py --data_dir=/data/StanfordCars --set=train --output_path=stanford_cars_train.tfrecord
(tf) $ python create_stanford_cars_tf_record.py --data_dir=/data/StanfordCars --set=test --output_path=stanford_cars_test.tfrecord
(tf) $ python create_stanford_cars_label_map.py
```

(Optional) Test create of tfrecords by dumping the data to a folder. Filename is currently hard-coded in the python script.
```
(tf) $ python dump.py
```

## TensorBoard screenshot

![tensorboard evaluation](tensorboard_stanford_cars.png)


## References

```
[0] TensorFlow Object Detection API, https://github.com/tensorflow/models/tree/master/research/object_detection
[1] Stanford Cars, https://ai.stanford.edu/~jkrause/cars/car_dataset.html
[2] Raccoon Detector, https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
[3] Toy Detector, https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
```

