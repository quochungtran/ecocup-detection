#!/usr/bin/env python3


import os 
import pandas as pd 
import numpy as np
from sklearn import model_selection
import ast
import shutil, sys  
import os 
import pandas as pd 
import numpy as np
from sklearn import model_selection
import cv2
import torch
from PIL import Image

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)



def process_data(data, data_type="train"):

    IMAGE_PATH = os.path.join(YOLO_DATA_PATH, f"images/{data_type}" )
    LABEL_PATH = os.path.join(YOLO_DATA_PATH, f"labels/{data_type}" )
    
    
    for id in range(data.shape[0]):
        yolo_data = []
        row = data.iloc[id]
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']

        img_name = row['name_image'][:-4]

        x_center = x + h/2
        y_center = y + w/2
        
        im = cv2.imread(os.path.join(DATA_PATH, f"images/pos/{img_name}.jpg"))
        (H,W) = im.shape[0], im.shape[1]
        x_center /= H
        y_center /= W
        w /= W
        h /= H

        yolo_data.append([0,y_center, x_center, w, h])
        
        yolo_data_convert_str = [str(x) for x in yolo_data] 

        file_txt_path = os.path.join(LABEL_PATH, f"{img_name}.txt")
        np.savetxt(
                file_txt_path,
                yolo_data,
                fmt = ["%d", "%f", "%f", "%f", "%f"]
            )
            
        shutil.copyfile(
            os.path.join(DATA_PATH, f"images/pos/{img_name}.jpg"),
            os.path.join(YOLO_DATA_PATH, f"images/{data_type}/{img_name}.jpg")
        )


if __name__ == "__main__":

    
    os.system("git clone https://github.com/ultralytics/yolov5")
    os.system("pip install -qr yolov5/requirements.txt")

    #---------------------------------------  Initialize structure of customed data format for yolo pretrained model --------------------------------------- 
    


    DATA_PATH      = r"../dataset-main/train"
    YOLO_DATA_PATH = r"yolo_data"
    LABEL_CSV_PATH = r"../dataset-main/train/labels_csv"
    
    if (not os.path.exists(os.path.join(YOLO_DATA_PATH, "images/train"))) :
        os.makedirs(os.path.join(YOLO_DATA_PATH, "images/train"))
    
    if (not os.path.exists(os.path.join(YOLO_DATA_PATH, "images/validation"))) :  
        os.makedirs(os.path.join(YOLO_DATA_PATH, "images/validation"))

    if (not os.path.exists(os.path.join(YOLO_DATA_PATH, "labels/train"))) :  
        os.makedirs(os.path.join(YOLO_DATA_PATH, "labels/train"))
    
    if (not os.path.exists(os.path.join(YOLO_DATA_PATH, "labels/validation"))) :  
        os.makedirs(os.path.join(YOLO_DATA_PATH, "labels/validation"))


    ##--------------------------------------- create train and validation set  --------------------------------------- 

    labels = pd.DataFrame()

    for dir1 in os.listdir(LABEL_CSV_PATH):
        csv_file_target = os.path.join(LABEL_CSV_PATH, dir1)
        df = pd.read_csv(csv_file_target, header= None)
        df.insert(0, "name_image", dir1[:-4] + '.jpg') 
        labels = labels.append(df, ignore_index=True)


    columns_names = ["name_image", "x", "y", "h", "w", "diff"]
    labels.columns = columns_names

    imgs_train, img_valid = model_selection.train_test_split(
        np.unique(labels['name_image']),
        test_size = 0.1,
        random_state = 42,
        shuffle=True
    )

    df_train = labels[labels['name_image'].isin(imgs_train)].reset_index(drop=True)
    df_valid = labels[labels['name_image'].isin(img_valid)].reset_index(drop=True)

    process_data(df_train, data_type="train")
    process_data(df_valid, data_type="validation")


    #---------------------------------------  training model --------------------------------------- 

    os.system("pip install -qr ./yolov5/requirements.txt")
    os.system("python3 ./yolov5/detect.py --weights  yolov5s.pt yolov5l.pt  yolov5m.pt yolov5x.pt")
    # os.system("python3 yolov5/train.py --img 1024 --batch 8 --epochs 100 --data ecocup.yaml --weights yolov5s.pt --cache")
    # after training above best weight is saved in best.pt

    #---------------------------------------  export results --------------------------------------- 

    os.system("python3 ./yolov5/detect.py --source ../dataset-main/test/ --weights best.pt")


    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
    DATA_PATH = "../dataset-main/test/"

    imgs      = []
    img_files = []
    data_res  = []

    for file in os.listdir(DATA_PATH):
      img = cv2.imread(os.path.join(DATA_PATH, file))
      results = model(img)
      preds = results.pandas().xyxy

      for id in range(len(preds[0].index.values)):
        x_min = int(preds[0]['xmin'].values[id])
        y_min = int(preds[0]['ymin'].values[id])
        x_max = int(preds[0]['xmax'].values[id]) 
        y_max = int(preds[0]['ymax'].values[id])

        w = x_max - x_min
        h = y_max - y_min

        s = preds[0]['confidence'].values[id]

        d = [file, y_min, x_min, h, w, s]
        data_res.append(d)


    columns_names = ["name_image", "x", "y", "h", "w", "s"]
    results_yolov5 = pd.DataFrame(columns=columns_names,
                        data=data_res)
    results_yolov5.to_csv('results_yolov5.csv', index=False, header=False)



