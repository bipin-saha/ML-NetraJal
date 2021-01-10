import cv2
import numpy as np
import dlib
from math import hypot
import time
import csv
from playsound import playsound

#AGE,SEX(MALE/FEMALE),EYE HOR, EYE VIR,IDENTITY

#Calibration Board
calibration_board = np.zeros((1080, 1920, 3), np.uint8)
calibration_board[:] = (255, 255, 255)
#Edgepoints
#cv2.circle(calibration_board,(0,0),25,(0,0,255),-1)
#cv2.circle(calibration_board,(1520,0),25,(0,0,255),-1)
#cv2.circle(calibration_board,(0,822),25,(0,0,255),-1)
#cv2.circle(calibration_board,(1520,822),25,(0,0,255),-1)
#cv2.circle(calibration_board,(1520,411),25,(0,0,255),-1)
#cv2.circle(calibration_board,(0,411),25,(0,0,255),-1)
#cv2.circle(calibration_board,(760,822),25,(0,0,255),-1)
#Midpoints
cv2.circle(calibration_board,(760,411),10,(0,0,255),-1)
cv2.circle(calibration_board,(380,205),10,(0,0,255),-1)
cv2.circle(calibration_board,(1140,205),10,(0,0,255),-1)
cv2.circle(calibration_board,(380,615),10,(0,0,255),-1)
cv2.circle(calibration_board,(1140,615),10,(0,0,255),-1)

#time.sleep(2)




###################            SETTINGS             ###################

#TOP-LEFT,LEFT,DOWN-LEFT,DOWN,DOWN-RIGHT,RIGHT,TOP-RIGHT,TOP,CENTER,TML,DML,TMR,DMR


gaze_lable = "TMR"
light = "DAYLIGHT-Mobile Flash"
device = "ASUS 530FN"
contributor = "ZZ"
contributor_age = "22"
contributor_sex = "M"
eye_horizontal_length = ""            #in cm
eye_verticle_length = ""               #in cm
identiy = "BANGLADESHI"

filename = "C:\\Users\\ASUS\\OneDrive\\Desktop\\NetraJal\\ML NetraJal\\Data Acquisition 2.2b\\UserProfile\\"+contributor+gaze_lable+"Gaze.csv"
video_filename = "C:\\Users\\ASUS\\OneDrive\\Desktop\\NetraJal\\ML NetraJal\\Data Acquisition 2.2b\\VideoFiles\\"+contributor+gaze_lable+"Gaze.avi"
cap = cv2.VideoCapture(0)
#address = "http://192.168.0.101:8080/video"
#cap.open(address)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename,fourcc, 20.0, (640,480))

###################            SETTINGS             ###################

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    #gray_eye_2 = cv2.resize(gray_eye, None, fx=5, fy=5)
    #cv2.imshow("Eye",gray_eye_2)
    #cv2.imwrite("ABC.jpg",gray_eye_2)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    vertical_left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    vertical_left_side_white = cv2.countNonZero(vertical_left_side_threshold)
    
    horizontal_left_side_threshold = threshold_eye[0: int(height/2), 0: width]
    horizontal_left_side_white = cv2.countNonZero(horizontal_left_side_threshold)

    vertical_right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    vertical_right_side_white = cv2.countNonZero(vertical_right_side_threshold)
    
    horizontal_right_side_threshold = threshold_eye[int(height/2) : height, 0: width]
    horizontal_right_side_white = cv2.countNonZero(horizontal_right_side_threshold)

    if vertical_left_side_white == 0:
        gaze_ratio = 1
    elif vertical_right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = vertical_left_side_white / vertical_right_side_white
    return gaze_ratio,vertical_left_side_white,vertical_right_side_white,horizontal_left_side_white,horizontal_right_side_white
    

left_eye_left_white = []
left_eye_right_white = []
right_eye_left_white = []
right_eye_right_white = []

left_eye_up_white = []
left_eye_down_white = []
right_eye_up_white = []
right_eye_down_white = []


        
with open(filename,'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["left_eye_left_white","left_eye_right_white","left_eye_up_white","left_eye_down_white","right_eye_left_white","right_eye_right_white","right_eye_up_white","right_eye_down_white","gaze_lable","lighting_condition","contributor_name","age","sex","eye_horizontal_length","eye_vertical_length","identity","device"])
        
start_time = time.time()
timeout = 60
timeout = start_time+timeout
voice_flag = 0

while start_time<timeout:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if voice_flag == 0:
        playsound("C:\\Users\\ASUS\\OneDrive\\Desktop\\NetraJal\\ML NetraJal\\Data Acquisition 2.2b\\calibration.wav")
        voice_flag = 1  
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))


		# Gaze detection
        gaze_ratio_left_eye,left_eye_left_white,left_eye_right_white,left_eye_up_white,left_eye_down_white = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye,right_eye_left_white,right_eye_right_white,right_eye_up_white,right_eye_down_white = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        
        
        x_test = [(left_eye_left_white,left_eye_right_white,left_eye_up_white,left_eye_down_white,right_eye_left_white,right_eye_right_white,right_eye_up_white,right_eye_down_white)]
        #print(x_test)
        with open(filename,"a",newline='') as file:
            writer = csv.writer(file)
            writer.writerow([left_eye_left_white,left_eye_right_white,left_eye_up_white,left_eye_down_white,right_eye_left_white,right_eye_right_white,right_eye_up_white,right_eye_down_white,gaze_lable,light,contributor,contributor_age,contributor_sex,eye_horizontal_length,eye_verticle_length,identiy,device])
        
        if right_eye_right_white == 0:
            right_eye_right_white = 1
        if right_eye_left_white == 0:
            right_eye_left_white = 1
        if left_eye_left_white == 0:
            left_eye_left_white = 1
        
        right_ratio = left_eye_right_white/right_eye_right_white
        left_ratio = left_eye_left_white/right_eye_left_white
        extra_A = right_eye_right_white/left_eye_left_white
        
        if left_ratio == 0:
            left_ratio = 1
        
        up_ratio = right_ratio/left_ratio
        up_ratio_2 = ((left_eye_right_white/left_eye_left_white)/right_eye_left_white)/right_eye_right_white
        
        
        print("Left Eye : ", left_eye_left_white,left_eye_right_white,left_eye_up_white,left_eye_down_white)
        print("Right Eye : ", right_eye_left_white,right_eye_right_white,right_eye_up_white,right_eye_down_white)
        print("_____________________________")
        
            

        left_white = 0
        right_white = 0        
        
        #time_stamp = 0
       
        if 0.65 < right_ratio < 1.35 and extra_A > 1.5 :          #Night 2.5, Day 1.5
            #cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 0.7 < up_ratio < 1.3 and up_ratio_2 < 0.1:
            #cv2.putText(frame, "UPPER", (50, 100), font, 2, (255, 255, 255), 3)
            new_frame[:] = (255, 255, 255)
        else:
            new_frame[:] = (255, 0, 0)
            #cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            
        



    cv2.imshow("Frame", cv2.resize(frame,(320,240)))
    #cv2.imshow("New frame", new_frame)
    
    
    cv2.imshow("Calibration Board",calibration_board)
    out.write(frame)
    
    
    #end_time = time.time()
    #diff = end_time-start_time
    start_time = start_time+(time.time()-start_time)
    #print(time.time()-st)
    
    key = cv2.waitKey(30)
    if key == 27:
        break
    #time_stamp = time_stamp+1
    #cv2.putText(frame, gaze_ratio, (50, 100), font, 2, (0, 0, 255), 3)
    

cap.release()
cv2.destroyAllWindows()

#print(left_values,right_values,upper_values,center_values)
#left_max = max(left_values)
#left_min = min(left_values)

#print(left_max,left_min)


"""
RIGHT <=0.65
UPPER 0.66-1.49
CENTER 1.5-2.3
LEFT >2.3
"""