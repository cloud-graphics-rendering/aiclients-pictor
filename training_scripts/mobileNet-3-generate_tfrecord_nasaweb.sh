#training
python mobileNet-3-generate_tfrecord_nasaweb.py --image_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/train/ --csv_input=/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/train/tf.csv --output_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/train/train.record
#testing
python mobileNet-3-generate_tfrecord_nasaweb.py --image_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/test --csv_input=/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/test/tf.csv --output_path=/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/test/test.record
