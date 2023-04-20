import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import pytesseract
import os
from pymongo import MongoClient
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'
dir_list = os.listdir("carimage")
n=1
state_code={'AN': 'Andaman and Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar', 'CH': 'Chandigarh', 'DN': 'Dadra and Nagar Haveli', 'DD': 'Daman and Diu', 'DL': 'Delhi', 'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana', 'HP': 'Himachal Pradesh', 'JK': 'Jammu and Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala', 'LD': 'Lakshadweep', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OR': 'Orissa', 'PY': 'Pondicherry', 'PN': 'Punjab', 'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'TamilNadu', 'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'WB': 'West Bengal'}
client = MongoClient('mongodb://localhost:27017/')
mydatabase = client['Mydatabase']
mycollection=mydatabase['collection']
Lst=[]
for image_name in dir_list:
    try:
        record={}
        img = cv2.imread(f'carimage/{image_name}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
        edged = cv2.Canny(bfilter, 30, 200) 
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        text = pytesseract.image_to_string(cropped_image, config='--psm 11')
        text_lst=[]
        for i in text:
            if i.isalnum():
                text_lst.append(i)
        number_plate="".join(text_lst)
        record["id"]=number_plate
        if number_plate[0:2] in state_code:
            cv2.imwrite(f'indian_car/{number_plate}.png', img)
            record["status"]="indian"
            record['state']=state_code[number_plate[0:2]]
            x = mycollection.insert_one(record)
        else:
            cv2.imwrite(f'not_indian_car/{number_plate}.png', img)
            record["status"]="Not_Indian"
            record['state']="Not_define"
            x = mycollection.insert_one(record)
    except:
        pass






