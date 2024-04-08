from bs4 import BeautifulSoup
import os
import shutil
from IPython.display import clear_output

status_dic = {
    'person':0,
    'bicycle':1,
    'car':2,
    "motorcycle":3,
    "bus":5,
    "truck":7,
    "boat":8,
    "cone":80,
    "hole":81,
    "jersey":82,
    "pole":83,
}             #用dictionary 記錄label的名稱

def getYoloFormat(filename, label_path, img_path, yolo_path, newname):    
    with open(label_path+filename, 'r') as f:        
        soup = BeautifulSoup(f.read(), 'xml')
        imgname = soup.select_one('filename').text         #讀取xml
        image_w = soup.select_one('width').text
        image_h = soup.select_one('height').text
        ary = []
        for obj in soup.select('object'):                  #取出xmin, xmax, ymin, ymax及name
            xmin = int(obj.select_one('xmin').text)          #並且用status_dictionary 來轉換name，good =>2
            xmax = int(obj.select_one('xmax').text)
            ymin = int(obj.select_one('ymin').text)
            ymax = int(obj.select_one('ymax').text)            
            objclass = status_dic.get(obj.select_one('name').text)

            x = (xmin + (xmax-xmin)/2) * 1.0 / float(image_w)    #YOLO吃的參數檔有固定的格式
            y = (ymin + (ymax-ymin)/2) * 1.0 / float(image_h)    #先照YOLO的格式訂好x,y,w,h
            w = (xmax-xmin) * 1.0 / float(image_w)
            h = (ymax-ymin) * 1.0 / float(image_h)
            ary.append(' '.join([str(objclass), str(x), str(y), str(w), str(h)]))
      
        if os.path.exists(img_path+imgname+'.jpg'):                              # 圖片本來在image裡面，把圖片移到yolo資料夾下    
            shutil.copyfile(img_path+imgname+'.jpg', yolo_path+newname+'.jpg')     #同時把yolo參數檔寫到yolo之下
            with open(yolo_path+newname+'.txt', 'w') as f:
                f.write('\n'.join(ary))            
        elif os.path.exists(img_path+imgname):  #有的labelImg名稱有自動加上.jpg                           
            shutil.copyfile(img_path+imgname, yolo_path+newname+'.jpg')
            with open(yolo_path+newname+'.txt', 'w') as f:
                f.write('\n'.join(ary))
                
def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text,end="\r")
    
    

labelpath = '/home/jim93073/coco/test_coco_format/'           #設定路徑
imgpath = '/home/jim93073/coco/test_coco_format/'
yolopath = '/home/jim93073/coco/test_yolo_person/'
ary = []
total_progress = len(os.listdir(labelpath))
progress = 0
for idx, f in enumerate(os.listdir(labelpath)):   #透過getYoloFormat將圖像和參數檔全部寫到YOLO下
    progress += 1
    try:
        if f.split('.')[1] == 'xml':
            getYoloFormat(f, labelpath, imgpath, yolopath, str(idx))
    except Exception as e:
        print(e)
    update_progress(progress/total_progress)