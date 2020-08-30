#training
python mobileNet-3-generate_tfrecord_inmind.py --image_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/inmind/train --csv_input=/media/lty/newspace/BenchmarkFrameWork/data-sets/inmind/train/tf.csv --output_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/inmind/train/train.record

#testing
python mobileNet-3-generate_tfrecord_inmind.py --image_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/inmind/test --csv_input=/media/lty/newspace/BenchmarkFrameWork/data-sets/inmind/test/tf.csv --output_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/inmind/test/test.record
