import os
import cv2
import tensorflow as tf

# TODO input tfrecord and output path
tfrecords_filename = 'stanford_cars_test.tfrecord'
output_path = './temp'

#get the number of records in the tfrecord file
c = 0
for record in tf.python_io.tf_record_iterator(tfrecords_filename):
    c += 1

f = tfrecords_filename
#logfile.write(" {} : {}".format(f, c))
#logfile.flush()
print("going to restore {} files from {}".format(c,f))

tf.reset_default_graph()

# here a path to tfrecords file as list
fq = tf.train.string_input_producer([tfrecords_filename], num_epochs=c)
reader = tf.TFRecordReader()
_, v = reader.read(fq)
fk = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/object/class/text': tf.FixedLenFeature([], tf.string, default_value=''),
    'image/object/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'image/filename': tf.FixedLenFeature([], tf.string, default_value='')
    }

ex = tf.parse_single_example(v, fk)
image = tf.image.decode_jpeg(ex['image/encoded'], dct_method='INTEGER_ACCURATE')
text = tf.cast(ex['image/object/class/text'], tf.string)
label = tf.cast(ex['image/object/class/label'], tf.int64)
fileName = tf.cast(ex['image/filename'], tf.string)
# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # set the number of images in your tfrecords file
    num_images=c
    print("going to restore {} files from {}".format(num_images, f))
    for i in range(num_images):

        im_,fName,text_,label_ = sess.run([image,fileName,text,label])
	print(fName, text_, label_)

        savePath=os.path.join(output_path,text_)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        base = os.path.basename(fName)
        fName_=os.path.join(savePath, base)

	print('saving {} to {}'.format(base, savePath))

        # change the image save path here
        cv2.imwrite(fName_ , im_)


    coord.request_stop()
    coord.join(threads)
