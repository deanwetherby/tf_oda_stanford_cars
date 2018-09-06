import os
import cv2
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('input_file','','tfrecord to dump')
flags.DEFINE_string('output_path','','Output folder to put images in associated class folders')
FLAGS = flags.FLAGS


def dump_records(tfrecords_filename, output_path):

  #get the number of records in the tfrecord file
  c = 0
  for record in tf.python_io.tf_record_iterator(tfrecords_filename):
      c += 1
  
  print("going to restore {} files from {}".format(c,tfrecords_filename))
  
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
      print("going to restore {} files from {}".format(num_images, tfrecords_filename))
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


def main():
  tfrecord = FLAGS.input_file
  output_path = FLAGS.output_path
  dump_records(tfrecord, output_path)


if __name__ == '__main__':
  tf.app.run()
