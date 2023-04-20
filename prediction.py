from ROI import *
import os
import LMTRP
import joblib
import numpy as np
import cv2
import glob
from PIL import Image 
def check():
# đường dẫn tới thư mục chứa các ảnh
    path_out_img = './ROI1'
    # Xóa toàn bộ tệp tin ảnh trong thư mục path_out_img

    file_list = glob.glob(os.path.join(path_out_img, '*.bmp'))
    for file_path in file_list:
        os.remove(file_path)


    roiImageFromHand(path_out_img, option=2, cap=cv2.VideoCapture(0))
    # lấy danh sách các tệp tin trong folder
    # file_list = os.listdir(path_out_img)

    # lọc ra danh sách các ảnh trong folder
    image_list = glob.glob(os.path.join(path_out_img, '*.bmp'))

    # load mô hình đã được train
    recognizer = joblib.load('./data/classifiers/user_classifier.joblib')
    pred = 0
    print_flag = True

    results = []
    confidence_scores = []
    for img in image_list:
        img= cv2.imread(img)
        img = cv2.resize(img, (64, 64))
        feature = LMTRP.LMTRP_process(img)
        
        # feature = LMTRP.LMTRP_process(img)
        feature = feature.reshape(1, -1)
        decision = recognizer.decision_function(feature)
        confidence = 1 / (1 + np.exp(-decision))
        predict = recognizer.predict(feature)
        # print(confidence)
        
        if predict[0]==1:
            pred = pred + 1
            confidence_scores.append(confidence)
            # text = "Nhan"
            # print(text)
        if print_flag:
            print("Prediction............!")
            print_flag = False
        # results.append(pred,confidence)


    sum=np.sum(confidence_scores)

    if pred>=5 and sum >=5: 
        print(np.sum(confidence_scores))
        return True
    else:
        print(np.sum(confidence_scores))
        return False


# if check():
#     print("pass")
# else:
#     print("notpass")




    #     print('User')
    #     # print(pred)
    #     print(np.sum(confidence_scores))
    #     Image1 = Image.open(f".\\2.png") 
    #     Image1copy = Image1.copy() 
    #     Image2=Image.open(f".\\tick.png")
    #     Image2copy = Image2.copy()
    #     Image1copy.paste(Image2copy, (195, 114)) 
    #     Image1copy.save("end.png") 
    #     frame = cv2.imread("end.png", 1)
    #     cv2.imshow("Result",frame)
    #     cv2.waitKey(2000)
    #     cv2.destroyAllWindows()
    # else :
    #     print('Unknown')
    #     # print(pred)
    #     print(np.sum(confidence_scores))
    #     frame1=cv2.imread("frame1.png",1)
    #     cv2.imshow("Access denied.",frame1)
    #     cv2.waitKey(2000)
    #     cv2.destroyAllWindows()


