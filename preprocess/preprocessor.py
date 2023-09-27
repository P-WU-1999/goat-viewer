import math
import os
import sys

import numpy as np
import cv2
from ultralytics import YOLO

# Path to the models
M_DETECT_PATH = '../models/YOLO/detect/goat_face_features-1.pt'
M_SEGMENT_PATH = '../models/YOLO/segment/model-i.pt'

# Detect model classes:
# 0: face
# 1: eye
# 2: mouth (consider as nose)
# 3: ear  --not used

SOURCE_PATH = '../datasets/source'    # Path to the image folder
OUTPUT_PATH = '../datasets/preprocessout'     # Path to the output folder

# Load model
model_detect = YOLO(M_DETECT_PATH)
model_segment = YOLO(M_SEGMENT_PATH)

scale_size = 64        # Output image size, the resolution should be scale_size x scale_size
nose_offset = 0.10     # Position of mouth on the face that can pass the filter (1 = turn off validation)

# Calculate the angle of a line connect 2 points and horizontal line.
def calculateAngle(point_1, point_2):
    delta_x = point_2[0] - point_1[0]
    delta_y = point_2[1] - point_1[1]

    angle_rad = math.atan2(delta_y, delta_x)

    angle_rad = angle_rad % math.pi

    # Ensure it is the minimum angle
    if angle_rad > (math.pi / 2):
        angle_rad = angle_rad - math.pi

    return angle_rad

# Calculate the coordinates of the rotated point
def rotatePoints(point, center, angle):
    rotated_x = center[0] + (point[0] - center[0]) * math.cos(-angle) - (point[1] - center[1]) * math.sin(-angle)
    rotated_y = center[1] + (point[0] - center[0]) * math.sin(-angle) + (point[1] - center[1]) * math.cos(-angle)

    return [int(rotated_x), int(rotated_y)]

# Check the position of goat face features if is in the correct position
def checkGoat(det_data, seg_data):
    pos_bias = 50   # Parameter that control validation range
    face = 0
    eye = 0
    nose = 0

    if len(seg_data.boxes) == 1:
        for boxID in range(len(det_data)):
            cls = int(det_data[boxID].cls[0])  # Class: / 0 face / 1 eye / 2 nose /
            c = [int(det_data[boxID].xywh[0, 0]), int(det_data[boxID].xywh[0, 1])]  # Center of the box

            # Check if the position is in the face
            if not (seg_data.boxes[0].xyxy[0, 0] - pos_bias <= c[0] <= seg_data.boxes[0].xyxy[0, 2] + pos_bias and
                    seg_data.boxes[0].xyxy[0, 1] - pos_bias <= c[1] <= seg_data.boxes[0].xyxy[0, 3] + pos_bias):
                return False

            # Count face features
            if cls == 0:
                face += 1
                facedata = det_data[boxID]
            elif cls == 1:
                eye += 1
            elif cls == 2:
                nose += 1

        if face == 1 and eye == 2 and nose == 1:
            return facedata  # Only one goat in the image and it's face feature is clear

    return False

# Return the coordinate of face features
def faceFeature(data, type, number):
    count = 0
    feature = []
    for boxID in range(len(data)):
        t = int(data[boxID].cls[0])

        # If this box is the required feature
        if t == type:
            feature.append(data[boxID])
            count += 1

            if count == number:
                return feature

    print("Feature number error, please check input data!")
    sys.exit()

# Doing preprocess
def preProcess():
    # Get a list of all items (files and subfolders) within the specified folder
    items = os.listdir(SOURCE_PATH)
    print(SOURCE_PATH)
    print(items)

    # Separate files and subfolders
    subfolders = [item for item in items if os.path.isdir(os.path.join(SOURCE_PATH, item))]


    for subfolder in subfolders:

        source = SOURCE_PATH + '/' + subfolder
        save_path = OUTPUT_PATH + '/' + subfolder + '/'

        # Generator of Results objects
        segment_results = model_segment.predict(source, device=0, conf=0.75)
        detect_results = model_detect.predict(source, device=0, conf=0.65)

        count = 0
        imgID = 0

        # Process each image
        for imgID in range(len(detect_results)):
            imgpath = str(detect_results[imgID].path)
            boxdata = detect_results[imgID].boxes
            segdata = segment_results[imgID]

            goatface = checkGoat(boxdata, segdata)  # check if there is only one

            # If there is a valid goat face
            if goatface:
                image = cv2.imread(imgpath)
                # ------------------------------------------------------------------------------------ #
                # Convert the image to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # ------------------------------------------------------------------------------------ #
                # Doing segmentation
                coor = segdata.masks.xy[0]
                points = []
                for i in range(len(coor)):
                    temp = (int(coor[i][0]), int(coor[i][1]))
                    points.append(temp)

                # Create a mask for the polygon
                mask = np.zeros_like(image)
                cv2.fillPoly(mask, [np.array(points)], color=(255, 255, 255))

                # Copy the area under the polygon from the original image
                copied_area = cv2.bitwise_and(image, mask)

                # Fill the entire image with black
                image = np.ones_like(image) * 0
                # Fill the entire image with white
                # image = np.ones_like(image) * 255

                # Paste the copied area back onto the white image
                image = cv2.bitwise_and(image, cv2.bitwise_not(mask)) + copied_area
                # ------------------------------------------------------------------------------------ #
                # Eyes alignment
                eyes = faceFeature(boxdata, type=1, number=2)  # Find 2 eyes
                nose = faceFeature(boxdata, type=2, number=1)  # Find nose
                eye_center = [[int(eyes[0].xywh[0, 0]), int(eyes[0].xywh[0, 1])],
                              [int(eyes[1].xywh[0, 0]), int(eyes[1].xywh[0, 1])]]
                nose_center = [int(nose[0].xywh[0, 0]), int(nose[0].xywh[0, 1])]
                face_center = [int(goatface.xywh[0, 0]), int(goatface.xywh[0, 1])]

                # Rotate the image
                height, width = image.shape[:2]
                angle = calculateAngle(eye_center[0], eye_center[1])    # Calculate the angle of 2 eyes

                rotation_matrix = cv2.getRotationMatrix2D(face_center, math.degrees(angle),
                                                          1.0)  # Generate the rotation matrix
                image = cv2.warpAffine(image, rotation_matrix, (width, height))  # Perform the rotation
                # ------------------------------------------------------------------------------------ #
                # Frontal face filter

                # Get rotated points
                eye_center[0] = rotatePoints(eye_center[0], face_center, angle)
                eye_center[1] = rotatePoints(eye_center[1], face_center, angle)
                nose_center = rotatePoints(nose_center, face_center, angle)

                # Validate nose position
                face_width = int(goatface.xyxy[0, 2]) - int(goatface.xyxy[0, 0])
                face_height = int(goatface.xyxy[0, 3]) - int(goatface.xyxy[0, 1])
                ref_line_x = (eye_center[0][0] + eye_center[1][0]) // 2
                if abs(ref_line_x - nose_center[0]) <= face_width * (nose_offset / 2):
                    # ------------------------------------------------------------------------------------ #
                    # Crop the image

                    if face_height > face_width:
                        bias = (face_height - face_width) / 2
                        image = image[int(goatface.xyxy[0, 1]):int(goatface.xyxy[0, 3]),
                                int(goatface.xyxy[0, 0] - bias):int(goatface.xyxy[0, 2] + bias)]
                    else:
                        bias = (face_height - face_width) / 2
                        image = image[int(goatface.xyxy[0, 1] - bias):int(goatface.xyxy[0, 3] + bias),
                                int(goatface.xyxy[0, 0]):int(goatface.xyxy[0, 2])]

                    # ------------------------------------------------------------------------------------ #
                    # Scale the image

                    image = cv2.resize(image, (scale_size, scale_size))

                    # ------------------------------------------------------------------------------------ #
                    # Change contrast and brightness

                    brightness_factor = 0.7
                    contrast_factor = 1.5

                    # Convert image to floating point
                    image = image.astype(float)

                    # Adjust brightness
                    image = image * brightness_factor

                    # Adjust contrast
                    image = image * contrast_factor

                    # Clip the pixel values to the valid range [0, 255]
                    image = np.clip(image, 0, 255)

                    # Convert to uint8 type
                    image = image.astype(np.uint8)
                    # ------------------------------------------------------------------------------------ #
                    # Save the pre-processed image
                    count += 1
                    directory = os.path.dirname(save_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    cv2.imwrite(save_path + str(count) + '.jpg', image)

        print('Floder:' + save_path + ' Finished! total image number: ' + str(imgID + 1) + ' passed: ' + str(count))

if __name__ == '__main__':
    preProcess()
