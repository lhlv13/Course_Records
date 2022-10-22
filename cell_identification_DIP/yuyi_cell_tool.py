import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from datetime import datetime
import copy



# In[0]  
# =============================================================================
# 大量影像讀取、顯示、儲存、相同應用、影片
# =============================================================================
def imreadImgs(filepath,Output_type='list'):
    """ filepath = ~/imgfile ,   
        Output_type is list or dict"""
    try:
        imglist = os.listdir(filepath)
    except:       
        raise Exception('path is error!!')
    if Output_type=='list':
        output_list = []
        output_name_list = []
        for img in imglist:
            try:
                i = cv2.imread(filepath+'/'+img,1)
            except:
                print(img,"讀取錯誤")
      
            output_list.append(i)
            output_name_list.append(img)
        return output_list,output_name_list
    
    elif Output_type=='dict':
        output_dir = {}
        for img in imglist:
            try:
                i = cv2.imread(filepath+'/'+img,1)
            except:
                print(img,"讀取錯誤")
           
            output_dir[img] = i
        return output_dir

    print("Output_type 請輸入 list 或 dict")
    return None
 

def imshowImgs(Imgs):
    """  Imgs maybe be a list or dict """
    l = []
    d = {}
    if(type(Imgs)==type(l)):  ### list
        num = 1
        for img in Imgs:
            cv2.imshow(str(num),img)
            cv2.waitKey(0)
            num +=1
    elif(type(Imgs)==type(d)):### dict
        for imgKey in Imgs:
            cv2.imshow(imgKey,Imgs[imgKey])
            cv2.waitKey(0)
    
    
    cv2.destroyAllWindows()

def imwriteImgs(Imgs,outpath='./outputImg/',extension=".jpg"):
    """ Imgs is list or dict ,
        outpath is file path ,
        extension is .jpg or .png"""
    if os.path.isfile(outpath):
        raise FileExistsError("path dose not exist.")
        
    l = []
    d = {}
    if type(Imgs)==type(l):
        """ """
        num=0
        for i in Imgs:
            num +=1
            try:
                now = datetime.now()
                date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
                pat = outpath+'/'+date_time+str(num)+extension
                cv2.imwrite(pat,i)
            except:
                print(i+'is error')
        print("save finish!!")
    
    elif type(Imgs)==type(d):
        for i in Imgs:
            try:
                pat = outpath+'/'+i+extension
                cv2.imwrite(pat,Imgs[i])
            except:
                print(i+'is error')
        print("save finish!!")
    else:    
        print("Imgs is list or dict.")




# In[0.5]
# =============================================================================
# 好用的 滑桿
# =============================================================================

def bar(anyImg, bar_num, function, imgshow_function=None,
        bar_name_list=[],bar_min_list=[],bar_max_list=[],
        window_name='bar',window_size=(640,480),
        save_img = False):
    """ """
    ### 這函數可以把你對照片的調整變成不限制數量的滑桿 讓你可以靠調整滑桿拯救世界(時間)!!
    ### 
    ### anyImg 放上一張cv2.imread後的照片
    ### bar_num 是指需要滑桿數量 ， bar_num = bar_variable_list數量
    ### function 是你要對圖片做的處理函數， input必須是 一個影像及，一個參數的list
    ###################################### output是你想要顯現的影像list ，list大小可以不跟輸入不一樣
    ### imgshow_function 可以設定你要imshow的圖片們 輸入是list 不用輸出
    
    ### 其他參數可設定可不設定
       
    
    ##### 除錯用
    
    assert bar_num>=len(bar_name_list) , "bar function : bar_name_list can\'t more than bar_num"   
    assert bar_num>=len(bar_min_list) , "bar function : bar_min_list can\'t more than bar_num"   
    assert bar_num>=len(bar_max_list) , "bar function : bar_max_list can\'t more than bar_num"   
    
    if(bar_name_list==[]):
        bar_name_list = ['bar_'+str(i) for i in range(1,bar_num+1)]
    elif(len(bar_name_list)<bar_num):
        x = bar_num-len(bar_name_list)
        for i in range(x):
            bar_name_list.append('0')
    
    if(bar_min_list==[]):
        bar_min_list = [0 for i in range(bar_num)]
    elif(len(bar_min_list)<bar_num):
        x = bar_num-len(bar_min_list)
        for i in range(x):
            bar_min_list.append(0)
            
    if(bar_max_list==[]):
        bar_max_list = [255 for i in range(bar_num)]          
    elif(len(bar_max_list)<bar_num):
        x = bar_num-len(bar_max_list)
        for i in range(x):
            bar_max_list.append(255)
    
    try:
        os.makedirs('./outputImg')
    except FileExistsError:
        pass
    
    
    
    
    
   
        
    cv2.namedWindow(window_name)   ### 創建窗口 名字預設為 'bar'
    cv2.resizeWindow(window_name, window_size[0],window_size[1] );  ###窗口大小
    
    ### 創建滑動窗口在 bar窗口 內
    def createTrackbarfunc(x):
        pass
    for i in range(bar_num):
        cv2.createTrackbar(bar_name_list[i], window_name, bar_min_list[i], bar_max_list[i], createTrackbarfunc)

         ### 設置滑動窗口初始默認值
        cv2.setTrackbarPos(bar_name_list[i], window_name,0)




     ### 創造滑桿所需變數數量
    bar_variable_list = []
    for i in range(bar_num):
        bar_variable_list.append(0)
    
    bar_output_img_list = [anyImg]
    
    save_list = []
    #################################### 正題
    while True:
        barImg = anyImg.copy()
        
        ### 取得滑動窗口的值
        for i in range(bar_num):
            bar_variable_list[i] = cv2.getTrackbarPos(bar_name_list[i], window_name)
        
        ### 設定處理照片的函數  輸入list 輸出也是 list
        try:
            bar_output_img_list = function(barImg,bar_variable_list)
        except Exception:
            print('bar function : parameter>> function, has problem!!!')
            
            
        ### 顯現照片函數
        if imgshow_function == None:
            cv2.imshow('bar_img',bar_output_img_list[0])
            pass
        else:
            try:
                imgshow_function(bar_output_img_list)
            except Exception:
                print('bar function : parameter>> imgshow_function, has problem!!!')
                
                
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            return save_list
        ## 按下 s 或 enter 儲存照片
        elif k== ord('s') or k == 13:  ### 13是enter鍵
            if save_img:
                imwriteImgs(bar_output_img_list)
            print(bar_variable_list)
            save_list.append(copy.deepcopy(bar_variable_list))
            print(save_list)
            # now = datetime.now()
            # date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            # fn = './outputImg/'+date_time+'color_space.jpg'
            # cv2.imwrite(fn,barImg)
            # print("儲存成功!!")
            
    cv2.destroyAllWindows()

def color_space_set_lower_upper(colorimg,low_np,upper_np,isShow=False):
    """ """
    csImg = colorimg.copy()
              
    channels = cv2.split(csImg)        
    for i in range(3):
        min_bool = channels[i]>=low_np[i]
        max_bool = channels[i]<=upper_np[i]
        boo = min_bool&max_bool    ### 範圍內的所有值接True
        channels[i][:]=0
        channels[i][boo] = 0xFF    ### 將True改成255
        csImg[:,:,i] =  csImg[:,:,i]&channels[i]
    if isShow:
        cv2.imshow('color_space',csImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return csImg
def bgr(colorImg,isShow=False):
    (B,G,R) =cv2.split(colorImg)
    if(isShow==True):
        cv2.imshow("B",B)
        cv2.waitKey(0)
        cv2.imshow("G",G)
        cv2.waitKey(0)
        cv2.imshow("R",R)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return B,G,R

def bgr2color(B,G,R,isShow=True):
    mergeImg = cv2.merge([B,G,R])
    if(isShow==True):
        cv2.imshow("mergeImg",mergeImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mergeImg

def color2others(colorImg,mode=cv2.COLOR_BGR2GRAY,isShow=True):
    cvtImg = cv2.cvtColor(colorImg, mode)
    if(isShow==True):
        cv2.imshow("cvtImg",cvtImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return cvtImg


def words(anyimg,content,position,scale=0.9,color=(0,255,255),bold=1,isShow=0):
    """ cv2.putText(anyimg, content, position, cv2.FONT_HERSHEY_SIMPLEX,scale, color, bold, cv2.LINE_AA) """
    cv2.putText(anyimg, content, position, cv2.FONT_HERSHEY_SIMPLEX,scale, color, bold, cv2.LINE_AA)
    
    if isShow:
        cv2.imshow("words",anyimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def otsu(Img, isShow=True):
    """ ret, out = cv2.threshold(src,threshold,max,method) """
    otsuThresh, otsuImg = cv2.threshold(Img, 0, 255, cv2.THRESH_OTSU)
    if (isShow==True):
        print('otsuThreshold =',otsuThresh)
        cv2.imshow('otsuImg',otsuImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return otsuImg


    
def contour(colorImg,binaryImg,color=(0,0,255),\
            mode= cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE,\
            contourIdx=0,thickness=1,isShow=True):
    """ img, countours, hierarchy = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])  \\\\
        cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])"""
    
    parameters = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(parameters)==3):  ## cv2 3點多版返回三個參數
        contours = parameters[1]
    elif(len(parameters)==2):  ## cv2 4點多版返回兩個參數
        contours = parameters[0]
    contourImg = np.copy(colorImg)
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.drawContours(contourImg, [cnt], contourIdx, color, thickness)
    
    if(isShow==True):
        cv2.imshow('contourImg',contourImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contourImg,contours


def contour_area_choose(contours,limit='>1000',isText_Show=False):
    """ """
    output_list = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        d = str(area)+limit
        d = eval(d)
        if d:
            output_list.append(contours[i])
    if isText_Show:
        print("contours"+limit+'有 :',len(output_list),"個")
    return output_list

def poly_contour(colorImg,contours,color=(0,0,255),\
                 contourIdx=0,thickness=1,isShow=True):
    """  cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])"""

    contourImg = np.copy(colorImg)
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.drawContours(contourImg, [cnt], contourIdx, color, thickness)
    
    if(isShow==True):
        cv2.imshow('contourImg',contourImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contourImg    


#####################################
def calc_contour_feature(contours_list):
    """
    輸入 contours
    回傳: feature list
    """
    feature_list = list()
    for cont in contours_list:
        area = cv2.contourArea(cont)   ## 面積
        if area == 0:
            continue
        perimeter = cv2.arcLength(cont, closed=True)   ## 邊長
        bbox = cv2.boundingRect(cont)
        #print(bbox)
        bbox2 = cv2.minAreaRect(cont)          ### 這個的框框不一定是正的，可能斜的
        #print(bbox2)
        circle = cv2.minEnclosingCircle(cont)
        if len(cont) > 5:                             #畫橢圓一定要大於 5
            ellipes = cv2.fitEllipse(cont)
        else:
            ellipes = None
        #print(ellipes)
        # Moment
        M = cv2.moments(cont) ## return all moment of given contour
        if area != 0: ## same as M["m00"] !=0
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = (None, None)
        feature = (center, area, perimeter, bbox, bbox2, circle, ellipes)
        feature_list.append(feature)
    return feature_list

def draw_cross(img,feature_list,color=(0,255,0),length = 5,isShow=False):
    img_cross = img.copy()
    for f in feature_list: ## center is fearue[0]
        if f[0][0] is not None:
            x, y = f[0][0],f[0][1]
            print(x,y)
            crossImg = cv2.line(img_cross, (x-length,y), (x+length,y), color=color)
            crossImg = cv2.line(crossImg, (x,y-length), (x,y+length), color=color)
    if isShow:
        cv2.imshow("image with cross", crossImg)
        cv2.waitKey(0)
    return crossImg

def draw_center(img, feature_list, color = (255, 255, 255), radius = 5, isShow = True):
    img_center = img.copy()
    for f in feature_list: ## center is fearue[0]
        if f[0][0] is not None:
            img_center = cv2.circle(img_center, f[0],  radius, color, -1) ## -1 filled
    if isShow:
        cv2.imshow("image with center", img_center)
        cv2.waitKey(0)
    return img_center

def draw_bbox(img, feature_list, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox = img.copy()
    for f in feature_list: ## center is fearue[0]
        (x, y, w, h) = f[3]
        img_bbox = cv2.rectangle(img_bbox, (x, y), (x + w, y + h), color, width) ## -1 filled
    if isShow:
        cv2.imshow("image with bbox", img_bbox)
        cv2.waitKey(0)
    return img_bbox

def draw_bbox2(img, feature_list, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox2 = img.copy()
    for f in feature_list: ## center is fearue[0]
        box = np.int0(cv2.boxPoints (f[4]))  #–> int0會省略小數點後方的數字
        img_bbox = cv2.drawContours(img_bbox2, [box], -1, color, width)
    if isShow:
        cv2.imshow("image with bbox", img_bbox2)
        cv2.waitKey(0)
    return img_bbox2

def draw_minSCircle(img, feature_list, color = (0, 255, 0), width = 2, isShow = True):
    img_circle = img.copy()
    for f in feature_list: ## center is fearue[0]
        ((x, y), radius) = f[5]  #–> int0會省略小數點後方的數字
        img_circle = cv2.circle(img_circle, (int(x), int(y)), int(radius), color, width)
    if isShow:
        cv2.imshow("image with bbox", img_circle)
        cv2.waitKey(0)
    return img_circle


def polylines(anyImg,contour_np,color=[0,255,0],thickness=3,isClosed=True,isShow=False):
    """ cv2.polylines(image, pts=[points], isClosed=True, color=red_color, thickness=3) """
    polyImg = cv2.polylines(anyImg, [contour_np], isClosed=isClosed, color=color,thickness=thickness)
    
    if isShow:
        cv2.imshow('polylines_Img',polyImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return polyImg

#%%
# =============================================================================
#  ROI 
# =============================================================================
def roi(anyImg,mask,BGR=(1,1,1),isShow=False):
    """ you can create a image only Region Of Interesting
    \\ cv2.fillPoly(image, [points], BGRcolor) """
    bitImg = np.zeros_like(anyImg)  #建立與照片一樣大的全0陣列
    cv2.fillPoly(bitImg, [mask], BGR) #建立感興趣區域
    roiImg = anyImg*bitImg
    
    if(isShow==True):
        cv2.imshow('roiImg',roiImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return roiImg

def roi_many_contours(anyimg,contours):
    """ you can input many contour"""
    roi_out_img = anyimg.copy()
    roi_out_img[:,:,:]= 0
    for ro in contours:
        roiImg = roi(anyimg, ro,isShow=0)
        roi_out_img = roi_out_img|roiImg
        
    return roi_out_img


def dilate(img, kernel= None, iterations = 1, isShow=True):
    if kernel is None:
        kernel = np.ones((3,3),np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations = iterations)
    if isShow:
        # cv2.imshow("Orignal", img)
        # cv2.waitKey(0)
        cv2.imshow("Dilate", img_dilate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    return img_dilate

def erode(img, kernel= None, iterations = 1, isShow=True):
    if kernel is None:
        kernel = np.ones((3,3),np.uint8)
    img_erode = cv2.erode(img, kernel, iterations = iterations)
    if isShow:
        # cv2.imshow("Orignal", img)
        # cv2.waitKey(0)
        cv2.imshow("Erode", img_erode)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    return img_erode

def kernel_3kinds(shape=(3,3),mode=cv2.MORPH_RECT):
    """ kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,15)) """
    return cv2.getStructuringElement(mode, shape)

def morphology(anyImg,mode,kernel,isShow=True):
    """  mophaImg = cv2.morphologyEx(Img, mode, kernel) """
    
    mophaImg = cv2.morphologyEx(anyImg, mode, kernel)
    if (isShow==True):
        cv2.imshow("morphologyImg",mophaImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mophaImg



