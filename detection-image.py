import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import easyocr
import numpy as np
import warnings
import cv2
import gradio as gr

print (tf.__version__)

PATH_TO_SAVED_MODEL = "Trained_Models\\loc_model1\\saved_model" #Input the path where the model of detection and localization is
new_model = tf.keras.models.load_model("Trained_Models\\cnn_model") # load the cnn model for logo brand prediction
print('Loading model...', end='')
start_time = time.time()


detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt",
                                                                    use_display_name=True)



warnings.filterwarnings('ignore')   

def detect(image):    
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
        for key, value in detections.items()}

        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=.30,
        agnostic_mode=False)

        
        cv2.imwrite('logo.jpg', image)
        flag1 = False
        flag2 = False
                        
        im_height, im_width, im_chan = image.shape
        
        i = 0
        while(flag1==False or flag2==False):
            if detections['detection_classes'][i] == 1 and flag1 == False:
                
                ymin,xmin,ymax,xmax = detections['detection_boxes'][i]

                xmin*=im_width
                xmax*=im_width
                ymin*=im_height
                ymax*=im_height

                dim =(256,256)
                
                class_names = ['Hyundai','Lexus','Mazda','Mercedes','Opel','Skoda','Toyota','Volkswagen'] # due to lack of datasets , currently limited to  8 brands only
                img = cv2.imread('logo.jpg')
                crop_img = img[round(ymin):round(ymax), round(xmin):round(xmax)]
                crop_img = crop_img.copy()
                resized2 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
                confidence = round((detections['detection_scores'][i]*100),2)
                print('Detected logo with ',confidence,'% confidence')
                
                
                resized = np.expand_dims(resized2,0)
                prediction = new_model.predict(resized)
                confidence = round((np.max(prediction)*100),2)
                print('Predicted brand is',class_names[np.argmax(prediction)])
                print('with',confidence,'% confidence')  
                text_out = 'Predicted brand is '+ class_names[np.argmax(prediction)]                                                                  
                flag1 = True
                i += 1
                
            if detections['detection_classes'][i] == 2 and flag2 == False:
                ymin,xmin,ymax,xmax = detections['detection_boxes'][i]

                xmin*=im_width
                xmax*=im_width
                ymin*=im_height
                ymax*=im_height
                dim = (400,100)

                img = cv2.imread('logo.jpg')
                crop_img = img[round(ymin):round(ymax), round(xmin):round(xmax)]
                crop_img = crop_img.copy()
                resized1 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
                
                
                
                reader = easyocr.Reader(['en'])
                ocr_result = reader.readtext(resized1)
                list = []
                for result in ocr_result:
                    length = np.sum(np.subtract(result[0][1],result[0][0]))
                    width = np.sum(np.subtract(result[0][2],result[0][1]))
                    threshold = (length*width)/30000
                    if(threshold>0.6): list.append(result[1])
                str1 = " "   
                str1 = str1.join(list)    
                text_out2 = 'License registration number is : ' + str1
                
                confidence1 = round((detections['detection_scores'][i]*100),2)
                flag2 = True
                i+=1
                
            if flag1 == True and flag2 == True:
                return (image_np_with_detections,resized1,text_out,text_out2)

            

    
 
   
iface = gr.Interface(detect,gr.inputs.Image(source="upload"),["image","image","text","text"])   
iface.launch(share=True)