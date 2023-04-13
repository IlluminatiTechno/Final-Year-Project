import cv2
import pyttsx3

## Initialize the text-to-speech engine
engine = pyttsx3.init()

##OPENCV DNN
net = cv2.dnn.readNet("C:/Users/Anirban/Documents/BAS-OPENCVn/BAS-OPENCVn/dnn_model/yolov4.weights", "C:/Users/Anirban/Documents/BAS-OPENCVn/BAS-OPENCVn/dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)


## LOAD CLASS LISTS
classes = []  ##empty list of python
file_name = r"C:\Users\Anirban\Documents\BAS-OPENCVn\BAS-OPENCVn\dnn_model\classes.txt"

with open(file_name, "rt") as fpt:
    for class_name in fpt.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

## INITIALIZE CAMERA
cap = cv2.VideoCapture(0)

## CHANGING VIDEO FEED RESOLUTION
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ## GET FRAMES
    ret,frame = cap.read()

    ##OBJECT DETECTION
    objects = []
    (class_ids, scores, bboxes)= model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x,y,w,h) = bbox
        class_name = classes[class_id]

        ## Check if the object is on the left or right
        if x < frame.shape[1]//2:
            side = "Left"
        else:
            side = "Right"

        ## Calculate the relative distance
        dist = round((w*h)/(frame.shape[0]*frame.shape[1]), 2)

        ## Add the object name, distance, and direction to the list
        objects.append((class_name, dist, side))

        ## Draw the bounding box and label
        label = f"{class_name} ({side}, {dist})"
        cv2.putText(frame, label, (x, y-10), font, fontScale= font_scale, color = (200, 0, 50), thickness= 2)
        cv2.rectangle(frame, (x,y),(x+w, y+h), (200, 0, 50), 3)

    ## Sort the list of objects by distance and direction
    objects = sorted(objects, key=lambda x: (x[1], x[2]))

    ## Convert the sorted list into speech
    speech_text = ""
    for obj in objects:
        speech_text += f"{obj[0]} on the {obj[2]} ({obj[1]} meters away), "

    engine.say(speech_text)
    engine.runAndWait()

    cv2.imshow("Frame", frame)
    cv2.waitKey(500) # Add a delay of 500 milliseconds (0.5 seconds)
