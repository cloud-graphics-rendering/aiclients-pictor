#training
TRAIN_PATH=/media/lty/newspace/BenchmarkFrameWork/data-sets/redeclipse/train
python3 mobileNet-3-generate_tfrecord_redeclipse.py --image_path=$TRAIN_PATH --csv_input=$TRAIN_PATH/tf.csv --output_path=$TRAIN_PATH/train.record

#testing
#python mobileNet-3-generate_tfrecord_redeclipse.py --image_path=/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/bench-datasets/redeclipse/test --csv_input=/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/bench-datasets/redeclipse/test/tf.csv --output_path=/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/bench-datasets/redeclipse/test/test.record
