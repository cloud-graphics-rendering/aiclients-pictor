import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TOD_Universal(object):
    def __init__(self, model_dir, label_dir, num_class, pic_region):
        self.PATH_TO_CKPT 	= model_dir
        self.PATH_TO_LABELS 	= label_dir
        self.NUM_CLASSES 	= num_class
        self.category_index 	= self._load_label_map()
        self.region 		= pic_region
        self.im_width           = self.region['width']
        self.im_height          = self.region['height']

    def _load_model(self):
        detection_graph 	= tf.Graph()
        with detection_graph.as_default():
            od_graph_def	= tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map	= label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories	= label_map_util.convert_label_map_to_categories(label_map,max_num_classes=self.NUM_CLASSES,use_display_name=True)
        category_index	= label_map_util.create_category_index(categories)
        return category_index

    def detect_objects(self, image, detection_graph, sess):
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor	  = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes		  = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores		  = detection_graph.get_tensor_by_name('detection_scores:0')
        classes		  = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections	  = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
               [boxes, scores, classes, num_detections],
               feed_dict={image_tensor: image_np_expanded})
        im_width = self.im_width
        im_height = self.im_height
        i_num = 0
        for score in scores[0]:
            if score > 0.5:
                i_num+=1
        valid_boxes = boxes[0][0:i_num]
        valid_classes = list(classes[0])
        obj_classes = valid_classes[0:i_num] 
        obj_positions = []
        for box in valid_boxes:
            ymin, xmin, ymax, xmax = box
            [left, right, top, bottom] = [xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height]
            obj_positions.append([left,right,top,bottom])

        vis_util.visualize_boxes_and_labels_on_image_array(
               image,
               np.squeeze(boxes),
               np.squeeze(classes).astype(np.int32),\
               np.squeeze(scores),
               self.category_index,
               use_normalized_coordinates=True,
               line_thickness = 8)

        return obj_classes, obj_positions, image
