from sre_constants import SUCCESS
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import tensorflow as tf
import warnings
import cv2
import easyocr

print (tf.__version__)

PATH_TO_SAVED_MODEL = "D:\\Projects\\Detection\\Trained_Models\\loc_model1\\saved_model" #Input the path where the model of detection and localization is
new_model = tf.keras.models.load_model("D:\\Projects\\Detection\\Trained_Models\\cnn_model") # load the cnn model for logo brand prediction
print('Loading model...', end='')
start_time = time.time()


detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap("D:\\Projects\\Detection\\label_map.pbtxt",##Input the path of label_map.pbtxt
                                                                    use_display_name=True)




 
max1=0
max2=0
xmin1=0
xmin2=0
ymin1=0
ymin2=0
xmax1=0
xmax2=0
ymax1=0
ymax2=0
warnings.filterwarnings('ignore')   

cap = cv2.VideoCapture(0)

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
        
        im_height, im_width, im_chan = image_np.shape
        cv2.imshow('frame',image_np_with_detections)
        idx =0
        
        for i in detections['detection_scores']:
            
            if i<0.6: break
            if(i>0.6 and detections['detection_classes'][idx]==1):
                if detections['detection_scores'][idx] > max1: 
                    max1 = detections['detection_scores'][idx] 
                    ymin1,xmin1,ymax1,xmax1 = detections['detection_boxes'][idx]
                    xmin1*=im_width
                    xmax1*=im_width
                    ymin1*=im_height
                    ymax1*=im_height
                    cv2.imwrite('logo1.jpg', image_np)
                idx+=1
            if(i>0.6 and detections['detection_classes'][idx]==2):
                if detections['detection_scores'][idx] > max2: 
                    max2 = detections['detection_scores'][idx] 
                    ymin2,xmin2,ymax2,xmax2 = detections['detection_boxes'][idx]
                    xmin2*=im_width
                    xmax2*=im_width
                    ymin2*=im_height
                    ymax2*=im_height
                    cv2.imwrite('num.jpg', image_np)
                idx+=1
            
    else:
        break    
    key = cv2.waitKey(1) & 0xFF 
    if  key == ord('a'):
        break 

    if key == ord('q'):
            
        #cv2.imshow("image",image_np)
        
        dim =(256,256)
        
        class_names = ['Hyundai','Lexus','Mazda','Mercedes','Opel','Skoda','Toyota','Volkswagen'] # due to lack of datasets , currently limited to  8 brands only
        img = cv2.imread('logo1.jpg')
        crop_img = img[round(ymin1):round(ymax1), round(xmin1):round(xmax1)]
        crop_img = crop_img.copy()
        resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        
        
        
        
        resized = np.expand_dims(resized,0)
        prediction = new_model.predict(resized)
        confidence = round((np.max(prediction)*100),2)
        text_out = 'Predicted brand is '+ class_names[np.argmax(prediction)] + ' with ' + str(confidence) + '% confidence....'                                                                  
        
        

        dim = (400,100)

        img = cv2.imread('num.jpg')
        crop_img = img[round(ymin2):round(ymax2), round(xmin2):round(xmax2)]
        crop_img = crop_img.copy()
        resized1 = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        
        
        
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(resized1)
        list = []
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1],result[0][0]))
            width = np.sum(np.subtract(result[0][2],result[0][1]))
            threshold = (length*width)/30000
            if(threshold>0.2): list.append(result[1])
        str1 = " "   
        str1 = str1.join(list)    
        text_out2 = 'Recognized license registration number is : ' + str1
        
        
       
        print(text_out+'\n'+text_out2)
    
    if key == ord('w'):
                
        #cv2.imshow("image",image_np)
        cv2.imwrite('logo.jpg', image_np)
        flag1 = False
        flag2 = False
                        
        im_height, im_width, im_chan = image_np.shape
        
        idx =0
        for i in detections['detection_classes']:
            if i == 1 and flag1 == False:
                
                ymin,xmin,ymax,xmax = detections['detection_boxes'][idx]

                xmin*=im_width
                xmax*=im_width
                ymin*=im_height
                ymax*=im_height

                dim =(256,256)
                
                class_names = ['Hyundai','Lexus','Mazda','Mercedes','Opel','Skoda','Toyota','Volkswagen'] # due to lack of datasets , currently limited to  8 brands only
                img = cv2.imread('logo.jpg')
                crop_img = img[round(ymin):round(ymax), round(xmin):round(xmax)]
                crop_img = crop_img.copy()
                resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
                confidence = round((detections['detection_scores'][i]*100),2)
                #print('Detected logo with ',confidence,'% confidence')
                
                
                resized = np.expand_dims(resized,0)
                prediction = new_model.predict(resized)
                confidence = round((np.max(prediction)*100),2)
                #print('Predicted brand is',class_names[np.argmax(prediction)])
                #print('with',confidence,'% confidence')  
                text_out = 'Predicted brand is '+ class_names[np.argmax(prediction)] + ' with ' + str(confidence) + '% confidence....'                                                                  
                flag1 = True
                idx+=1
                
            if  i == 2 and flag2 == False:
                ymin,xmin,ymax,xmax = detections['detection_boxes'][idx]

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
                    threshold = (length*width)/40000
                    if(threshold>0.3): list.append(result[1])
                str1 = " "   
                str1 = str1.join(list)    
                text_out2 = 'Recognized license registration number is : ' + str1
                
                flag2 = True
                idx+=1
            if flag1 == True and flag2 == True:
               break
        print(text_out+'\n'+text_out2)           
        
     
        
        
    
cap.release()
cv2.destroyAllWindows()  