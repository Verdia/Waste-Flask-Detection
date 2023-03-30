import numpy as np
import cv2
import tensorflow as tf
from utils.ssd_mobilenet_utils import *
import time

def run_detection(image, interpreter):
    # Run model: start to detect
    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]['index'], image)
    # Invoke the interpreter.
    interpreter.invoke()

    # get results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)
        
    return out_scores, out_boxes, out_classes

def real_time_object_detection(interpreter, colors):
    camera = cv2.VideoCapture(2)

    match_detect = 0
    idx_before = 0

    panjang = 0
    lebar = 0

    Status = False
    lastStatus = False

    tipe_botol = 0

    start = time.time()
        
    cup_total = 0
    count_330 = 0
    count_600 = 0
    count_1500 = 0

    berat_total = 0

    while camera.isOpened():
        startFPS = time.time()
        ret, frame = camera.read() #peopledtct

        if ret:
            image_data = preprocess_image_for_tflite(frame, model_image_size=300)
            out_scores, out_boxes, out_classes = run_detection(image_data, interpreter)
            # Draw bounding boxes on the image file
            result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)            
            end = time.time()

            h, w, _ = frame.shape

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                
                # ssd_mobilenet
                ymin, xmin, ymax, xmax = box
                left, right, top, bottom = (xmin * w, xmax * w,
                                        ymin * h, ymax * h)
                
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
                right = min(w, np.floor(right + 0.5).astype('int32'))

                match_detect = c

                panjang = right - left
                lebar = bottom - top

            if match_detect == 44 or match_detect == 47:
                if match_detect == idx_before:
                    elapsed = time.time() - start

                    if match_detect == 44 and elapsed > 1 :
                        # Botol 330ml = 305 x 105
                        # Botol 600ml = 390 x 140
                        # Botol 1500ml = 

                        print("class: {}, p : {}, l : {}".format(predicted_class, panjang, lebar))

                        # 1 = 330, 2 = 600, 3 = 1500
                        ###################################
                        if panjang < 100 and lebar < 240:
                            tipe_botol = 1
                        elif (panjang > 250 and panjang < 310) and (lebar > 120 and lebar < 150):
                            tipe_botol = 2
                        elif panjang > 250 and lebar > 160:
                            tipe_botol = 3

                        Status = True
                            
                        print("class: {}, p : {}, l : {}, time: {}, tipe_botol : {}".format(predicted_class, panjang, lebar, int(elapsed), tipe_botol))
                    
                    elif match_detect == 47 and elapsed > 1 :
                        Status = True
                            
                        print("class: {}, p : {}, l : {}, time: {}, tipe : cup".format(predicted_class, panjang, lebar, int(elapsed)))
                    
                else :
                    start = time.time()
                    Status = False
            else:
            
                elapsedUnDetect = time.time() - start

                # print(elapsedUnDetect)

                if  elapsedUnDetect > 30:
                    start = time.time()

                    cup_total = 0
                    count_330 = 0
                    count_600 = 0
                    count_1500 = 0
                    berat_total = 0

                    jsonData = {
                        "Botol_330": count_330,
                        "Botol_600": count_600,
                        "Botol_1500": count_1500,
                        "Cup_total": cup_total,
                        "Berat_total": berat_total
                    }
                    jsonObj = json.dumps(jsonData, indent = 5)

                    with open("dataDetection.json", "w") as outfile:
                        outfile.write(jsonObj)
                    
                    Status = False

            idx_before = match_detect

            if match_detect == 44 :
                if(Status != lastStatus and Status == True) and tipe_botol == 1 :
                    count_330 = count_330 + 1
                    berat_total = berat_total + 0.5

                    jsonData = {
                        "Botol_330": count_330,
                        "Botol_600": count_600,
                        "Botol_1500": count_1500,
                        "Cup_total": cup_total,
                        "Berat_total": berat_total
                    }
                    jsonObj = json.dumps(jsonData, indent = 5)

                    with open("dataDetection.json", "w") as outfile:
                        outfile.write(jsonObj)

                elif (Status != lastStatus and Status == True) and tipe_botol == 2 :
                    count_600 = count_600 + 1
                    berat_total = berat_total + 1.8

                    jsonData = {
                        "Botol_330": count_330,
                        "Botol_600": count_600,
                        "Botol_1500": count_1500,
                        "Cup_total": cup_total,
                        "Berat_total": berat_total
                    }
                    jsonObj = json.dumps(jsonData, indent = 5)

                    with open("dataDetection.json", "w") as outfile:
                        outfile.write(jsonObj)
                        
                elif (Status != lastStatus and Status == True) and tipe_botol == 3 :
                    count_1500 = count_1500 + 1
                    berat_total = berat_total + 4

                    jsonData = {
                        "Botol_330": count_330,
                        "Botol_600": count_600,
                        "Botol_1500": count_1500,
                        "Cup_total": cup_total,
                        "Berat_total": berat_total
                    }
                    jsonObj = json.dumps(jsonData, indent = 5)

                    with open("dataDetection.json", "w") as outfile:
                        outfile.write(jsonObj)
                        
            elif match_detect == 47 :
                if(Status != lastStatus and Status == True):
                    cup_total = cup_total + 1
                    berat_total = berat_total + 0.5

                    jsonData = {
                        "Botol_330": count_330,
                        "Botol_600": count_600,
                        "Botol_1500": count_1500,
                        "Cup_total": cup_total,
                        "Berat_total": berat_total
                    }
                    jsonObj = json.dumps(jsonData, indent = 5)

                    with open("dataDetection.json", "w") as outfile:
                        outfile.write(jsonObj)
                        
            
            lastStatus = Status

            match_detect = 0

            # fps
            t = end - startFPS
            
            fps  = "Fps: {:.2f}".format(1 / t)
            cv2.putText(result, fps, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # print(count_all)

        cv2.imshow("Object detection - ssdlite_mobilenet_v2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model_data/ssdlite_mobilenet_v2.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
            
    # image_object_detection(interpreter, colors)
    real_time_object_detection(interpreter, colors)