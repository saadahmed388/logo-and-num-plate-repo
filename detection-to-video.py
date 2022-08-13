from sre_constants import SUCCESS
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import warnings
import cv2


print (tf.__version__)

PATH_TO_SAVED_MODEL = "Trained_Models\\loc_model1\\saved_model" #Input the path where the model of detection and localization is
new_model = tf.keras.models.load_model("Trained_Models\\cnn_model")
print('Loading model...', end='')
start_time = time.time()


detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt",##Input the path of label_map.pbtxt
                                                                    use_display_name=True)
warnings.filterwarnings('ignore')   


cap = cv2.VideoCapture("C:\\Users\\PC\\Downloads\\video.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('detected.avi', 
                    cv2.VideoWriter_fourcc(*'MJPG'),
                    20, size)

while True:
    success, image_np = cap.read()
    if success:
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}

        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=2,
            min_score_thresh=.50,
            agnostic_mode=False)

        cv2.imshow('frame',image_np_with_detections)
        result.write(image_np_with_detections)
        
    else:
        break    

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
cap.release()
result.release()
cv2.destroyAllWindows()     
    

