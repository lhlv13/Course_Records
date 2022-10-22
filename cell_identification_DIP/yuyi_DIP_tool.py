import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from datetime import datetime
import copy

#%%
# =============================================================================
#  除錯
# =============================================================================
import logging

def logging(filename='loggin.txt',level=logging.DEBUG):
    ### 下面這些使用後會記錄在loggin.txt檔
    ### logging.debug('\n\nlogging msg')
    ### logging.info('logging msg')
    ### logging.warning('logging msg')
    ### logging.error('logging msg')
    ### logging.critical('logging msg')
    logging.basicConfig(filename='loggin.txt',level=logging.DEBUG,format='%(asctime)s - %(levelname)s : %(message)s')

import  traceback

def bug_record(error,path='traceback.txt'):
    ### 使用在  try...
    ###except Exception as error:
    print("錯誤錯誤!!",str(error))
    with open(path,'a') as t:
        t.write((traceback.format_exc()))

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


def Imgs_deal(Imgs,function):
    """ """
    ### 可以大量處理照片，處理手法都一樣
    output_list = []
    for img in Imgs:        
        output_list.append(function(img))
    return output_list
       
def capture_video(filepath=0,function=None,title="frame"):
    ### 如果沒填參數就代表捕捉webcame攝影機
    ### function 要自己在外部寫哦 然後看以要什麼處理過程就放進去吧各種慮波!?
    ### function在設置時輸入輸出都是一張照片唷
    ### filepath 可以是影片路徑 不填就是攝影機
    videocapture = cv2.VideoCapture(filepath)
    
    if(videocapture.isOpened()==0):  ## 攝影機沒自動打開的話
        videocapture.open()          ## 讓攝影機打開
    ret, frame = videocapture.read() ### ret 是True/False ， frame 是單張照片
    while ret:
        ret, frame = videocapture.read() ### ret 是True/False ， frame 是單張照片
        
        if function!=None:
            frame = function(frame)       ### 這裡設定自己要處理的函數 記得回傳照片
            
        cv2.imshow(title,frame)
        
        ### 設定按鍵功能
        key = cv2.waitKey(30) &0xFF
        if key ==ord('q') or key==27:  #按q键退出
            break
        elif key == ord('s') or key == 13:   ### 13是enter鍵
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            fn = './img/'+date_time+'img.jpg'
            cv2.imwrite(fn, frame)
            print("截圖成功")
    
    videocapture.release()
    cv2.destroyAllWindows()


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

# In[1]
# =============================================================================
# 找座標點用
# =============================================================================
def coordinate_pyplot(anyimg):
    """can show a image by pyplot and this image have shown coordinate position """
    ### 使用前要先 import matplotlib.pyplot as plt
    ### 這個顯示的圖可以顯示像素座標位置 很好用
    copyImg = anyimg.copy()
    copyImg = copyImg[:,:,::-1]
    plt.imshow(copyImg)

def coordinate(img):
    """ coordinate of every pixel"""
    #目前這個函數執行後不會退出 還要想辦法XD 但可以印出座標
    #在圖片點滑鼠左鍵就可以顯示該點座標
    #在圖片上點滑鼠右鍵可以刪除串列內最後一點座標
    coor = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: ## 按下左鍵
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            coor.append([x,y])
            print(coor)
            cv2.imshow("image", img)
            cv2.waitKey(0)
        if event == cv2.EVENT_RBUTTONDOWN:  ## 按下右鍵 取消前一個座標
            coor.pop()
            print(coor)
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

#%%
# =============================================================================
#  通道分割、通道合併、色彩轉換
# =============================================================================

def bgr(colorImg,isShow=False):
    ### 把彩色圖片分割成B G R三通道
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
    ### 將放上來的三張灰階圖片合併成一張彩色圖片
    mergeImg = cv2.merge([B,G,R])
    if(isShow==True):
        cv2.imshow("mergeImg",mergeImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mergeImg

def color2others(colorImg,mode=cv2.COLOR_BGR2GRAY,isShow=True):
    ### 將圖片轉其他色彩模式 預設為彩色轉灰階
    ### 轉hsv : cv2.COLOR_BGR2HSV
    ### 還有多種樣式 任君挑選  打上cv2.COLOR_BGR2 會有很多選擇
    cvtImg = cv2.cvtColor(colorImg, mode)
    if(isShow==True):
        cv2.imshow("cvtImg",cvtImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return cvtImg

#%%
# =============================================================================
# 圖片增加文字、將兩張圖片合再一起
# =============================================================================



def words(anyimg,content,position,scale=0.9,color=(0,255,255),bold=1,isShow=0):
    """ cv2.putText(anyimg, content, position, cv2.FONT_HERSHEY_SIMPLEX,scale, color, bold, cv2.LINE_AA) """
    ### 沒有回傳值，文字會直接儲存進圖像
    ### anyimg 可以彩色 以下以彩色為例
    ### content 是字串 輸入你要的文字印在圖上
    ### 設定你要印的文字的左下角座標 例如彩色圖片也是 (10,40)，這裡座標跟數學課的座標圖是一樣的(x,y)
    ################## 下面不一定要設置
    ### scale 是文字的大小 可以是浮點數
    ### color就是顏色啦 (B,G,R)
    ### bold 是線條的粗度
    cv2.putText(anyimg, content, position, cv2.FONT_HERSHEY_SIMPLEX,scale, color, bold, cv2.LINE_AA)
    
    if isShow:
        cv2.imshow("words",anyimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def concate_2img(colorimg1,colorimg2,mode=1,isShow=0):
    """ grayImg = np.concatenate((grayimg1, grayimg2), axis=mode)"""
    ### 可以將兩張圖片融合成一張
    ### mode = 0  是上下連接 
    ### mode = 1  是左右連接
    if mode ==0:
        assert colorimg1.shape[1]==colorimg2.shape[1], "concate_2img function : shape[1] need to same!!"
    if mode ==1:
        assert colorimg1.shape[0]==colorimg2.shape[0], "concate_2img function : shape[0] need to same!!"
    b1,g1,r1 = cv2.split(colorimg1)
    b2,g2,r2 = cv2.split(colorimg2)
    
    B = np.concatenate((b1, b2), axis=mode) 
    G = np.concatenate((g1, g2), axis=mode) 
    R = np.concatenate((r1, r2), axis=mode) 
    
    concateImg = cv2.merge([B,G,R])
    if isShow:
        cv2.imshow("concate_2img",concateImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return concateImg
        
#%%
def  convert_scale_abs(anyImg,alpha=1,beta=0 ,dst=0, isShow=True):
    """ Img = def convertScaleAbs(src, dst=None, alpha=None, beta=None) """
    ### 將影像變成0~255的範圍
    ### 調整alpha beta 會讓螢幕變亮變暗的樣子 應該就是景深!?
    ### 假設我們想讓深度鏡頭8m距離内的深度被顯示，>8m的與8m的颜色顯示相同，
    ### 那 alpha=255/(8*10^3)≈0.03
    ### alpha 是倍乘因子  beta 是偏移量
    uint8Img =  cv2.convertScaleAbs(anyImg,dst,alpha,beta)
    if (isShow==True):
        cv2.imshow('uint8Img',uint8Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return uint8Img

def add_weighted(Img1,opennessImg1,Img2,opennessImg2,brightness=100,isShow=True):
    """  cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) → dst """
    ### opennessImg1 : 第一張圖片的透明度，也就是權重
    ### brightness : gamma 可以說是調亮度
    if(Img1.ndim == Img2.ndim):
        addImg = cv2.addWeighted(Img1,opennessImg1,Img2,opennessImg2,brightness)
        if (isShow == True):
            cv2.imshow('addImg',addImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return addImg
    else:
        print("兩張照片大小不一致")
        return 0

def optimized():
    ### 在編譯時優化是被默認開啟的。因此OpenCV 運行的就是優化後的代碼，
    ### 如果你把優化關閉的話就只能執行低效的代碼了
    retval = cv2.useOptimized()  ### 查看優化是否被開啟
    cv2.setUseOptimized(True)    ### 開啟優化




# In[2]

def calcAndDrawHist(image, color):       
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])    
    print("Histogram: ", hist)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)   
    print(minVal, maxVal, minLoc, maxLoc) 
    print("Max Hist: ",hist[maxLoc[1]])
    histImg = np.zeros([256,256,3], np.uint8)    
    hpt = int(0.9* 256);           
    for h in range(256):    
        intensity = int(hist[h]*hpt/maxVal)   ## rescale histogram value into 0-255 
         ### cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
        cv2.line(histImg,(h,256), (h,256-intensity), color)           
    return histImg

# In[3]
# =============================================================================
#  直方圖
# =============================================================================
def histogram_show(img,title='Histogram'):   ##gray
    bins = np.arange(257)   
    item = img[:,:]  
    Y,X = np.histogram(item,bins)  ##X 是0~256 Y是 數量
    plt.bar(X[:-1], Y, align = 'center', width = 0.7)
    plt.xlabel('256 gray')
    plt.ylabel('numbers')
    plt.title(title)
    plt.show()  
    return 

    
def histogram_equalization(grayImg,isShow=True):
    """This function can increase image contrast,
    and let the original mass square map evenly distribute from 0 to 255"""
    
    if(grayImg.ndim==2):
        eqImg = cv2.equalizeHist(grayImg)
        histogram_show(eqImg)    #使用上面的直方圖
        
        if (isShow==True):
            cv2.imshow('histogram_equalization',eqImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("請輸入灰階圖片")
    return eqImg


    
def histogram_ranking_get(Img,rank,isShow=True):
    """ you can get top rank histogramImg"""
    ### rank 可以得到數量前幾名的灰階值 其他值為 0
    ### 輸出為Img
    rows, cols = Img.shape
    boolImg = np.zeros((rows,cols),dtype=np.uint8)
    
    dictHist = {}
    
    bins = np.arange(257)   
    item = Img[:,:]  
    Y,X = np.histogram(item,bins)  ##X 是數量 Y是 範圍
    for i in range(len(Y)):
        dictHist[X[i]] = Y[i]
    ranklist = sorted(dictHist.items(), key=lambda e:e[1], reverse=True)
    for j in range(rank):
        T = Img[:]==ranklist[j][0]
        boolImg[T] = 255
    # histogram_show(boolImg)
    if (isShow==True):
        print("IRank",ranklist[:rank])
        cv2.imshow("histogram_rank_get_Img",boolImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return boolImg
    
        
        
    


# In[4]

# =============================================================================
#  閥值
# =============================================================================


def threshold(Img, minthresh, maxthresh, method=cv2.THRESH_BINARY, isShow=True, allmethodShow=False):
    """  ret, out = cv2.threshold(src,threshold,max,method) 
        
         threshold : a min threshold
         max :  a max threshold
         method : cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV
    """
    ### minthresh : 最小門檻值
    ### maxthresh : 最大門檻值
    ### method :  cv2.THRESH_BINARY / THRESH_BINARY_INV / THRESH_TRUNC / THRESH_TOZERO / THRESH_TOZERO_INV
    ### 不同 method 會有不同效果哦
    
    if(allmethodShow != True):
        ### ret 是cv2.THRESH系列 返還的決定閥值
        ret,threshImg = cv2.threshold(Img, minthresh, maxthresh, method)
        if(isShow == True):
            cv2.imshow('thresholdImg',threshImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif(allmethodShow==True):
        met = (cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV)        
        name = ('THRESH_BINARY','THRESH_BINARY_INV','THRESH_TRUNC','THRESH_TOZERO','THRESH_TOZERO_INV')        
        for m in met:
            ret,threshAllImg = cv2.threshold(Img, minthresh, maxthresh, m)
            if(m==method):
                threshImg = threshAllImg
            
            cv2.imshow(str(name[m])+'Img',threshAllImg)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return threshImg
        
        

def average_intensity_threshold(Img):
    """ this function can ouput average intensity value of pixel  """
    ### 輸入一張照片 輸出為一個值
    ### 優點: 速度快 且閥值可隨光線改變自動調等
    ### 缺點: 物體突出之外型尖端將變得較平滑
    rows, cols = Img.shape
    thresh = np.sum(Img)/(rows*cols)
    print('average_intensity_threshold =',thresh,type(thresh))
    
    return thresh



def calc_sum_of_intensity(Img,thresh):
    
    lowThresh = Img[Img[:,:]<thresh]      #計算小於閥值的總強度
    lowThreshAvr = np.sum(lowThresh)/len(lowThresh)
    highThresh = Img[Img[:,:]>=thresh]      #計算小於閥值的總強度
    highThreshAvr = np.sum(highThresh)/len(highThresh)
    
    return lowThreshAvr,highThreshAvr
    
    

def modified_iterative_method(Img):
    """ ouput is a value of intensity  """
    ### 一開始 使用像素的平均強度當初使閥值
    ### 將閥值分成大於等於 和 小於 個別做平均
    ### 利用上面得到的兩個平均值 再做平均
    ### 與舊的做比較 直到迭代到值不會更動
    ### 輸出為一個閥值
    
    thresh = average_intensity_threshold(Img)  #得到平均閥值
    thresh_old = thresh
    
    while True:
        lowThreshAvr,highThreshAvr = calc_sum_of_intensity(Img,thresh)
        thresh = (lowThreshAvr+highThreshAvr)//2
        
        if(thresh == thresh_old):
            break
        else:
            thresh_old = thresh 
    thresh = thresh.astype(np.uint8)   
    return thresh
        

def otsu(Img, isShow=True):
    """ ret, out = cv2.threshold(src,threshold,max,method) """
    ### Otsu's非常適合於圖像灰度直方圖具有雙峰的情況
    ### 對於非雙峰圖像，可能並不是很好用
    otsuThresh, otsuImg = cv2.threshold(Img, 0, 255, cv2.THRESH_OTSU)
    if (isShow==True):
        print('otsuThreshold =',otsuThresh)
        cv2.imshow('otsuImg',otsuImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return otsuImg


def adaptive_threshold(Img,blockSize,Constant,  threshType=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                       method=cv2.THRESH_BINARY ,  maxval=255,isShow=True):
    """ dst = cv2.adaptiveThreshold(src, maxval, thresh_type, type, Block Size, C) """
    ### adaptive會分去塊運算 也就是blockSize(值越小計算當然越久)
    ### Constant 就是一個閥值的常數項，就試著設定吧XD
    ### threshType 有兩種 : cv2.ADAPTIVE_THRESH_GAUSSIAN_C  和 cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    ### method 有五種 : cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV
    adaptiveImg = cv2.adaptiveThreshold(Img,maxval,threshType,method,blockSize,Constant)
    if(isShow==True):
        cv2.imshow('adaptive_thresholdImg',adaptiveImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return adaptiveImg


#%%  不同kernel

# =============================================================================
#  捲積
# =============================================================================


def filters(name='',size=3):
    if name=='averaging':
        kernel = np.ones((size,size),dtype=np.float32)/(size**2)
    
    elif name =='laplacian':
        kernel = np.array([[0,1,0],
                           [1,-4,1],
                           [0,1,0]])
        
    elif name =='gaussian':
        gaussian_filter = cv2.getGaussianKernel(size, 0)
        kernel = gaussian_filter * gaussian_filter.transpose(1, 0) ## produec 2D matrix
        
        
    ### sobel 系列  
    elif name =='sobel_vertical':
        kernel = np.array([[1,0,-1],
                          [2,0,-2],
                          [1,0,-1]])
    elif name =='sobel_vertical2':
        kernel = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
    elif name =='sobel_horizon':
        kernel = np.array([[1,2,1],
                           [0,0,0],
                           [-1,-2,-1]])
    elif name =='sobel_horizon2':
        kernel = np.array([[-1,-2,-1],
                           [0,0,0],
                           [1,2,1]])
        
        #prewitt
    elif name =='vertical':
        kernel = np.array([[1,0,-1],
                           [1,0,-1],
                           [1,0,-1]])
    elif name =='vertical2':
        kernel = np.array([[-1,0,1],
                           [-1,0,1],
                           [-1,0,1]])
    elif name =='horizon':
        kernel = np.array([[1,1,1],
                           [0,0,0],
                           [-1,-1,-1]])
    elif name =='horizon2':
        kernel = np.array([[-1,-1,-1],
                          [0,0,0],
                          [1,1,1]])
    
    return kernel


#%%  捲積

def myconvolution(grayImg,kernel,isShow=True):
    """ convolusion(grayImg,kernel)
        grayImg : only gray image, ndim is 2
        kernel can put 3,5,7,9..."""   
    if(grayImg.ndim ==2):
        convImg = np.copy(grayImg)
        print(type(convImg))
        rows, cols = convImg.shape
        ker_rows, ker_cols = kernel.shape
        for r in range(int(ker_rows//2),int(rows-ker_rows/2)):
            for c in range(int(ker_cols//2),int(cols-ker_cols/2)):
                r_start = r - int(ker_rows//2)
                r_end = r + int(ker_rows//2) + 1  ##加 1 是因為串列最後一個不會計算ex: [1:d]
                c_start = c - int(ker_cols//2)
                c_end = c + int(ker_cols//2) + 1
                convImg[r,c] = np.sum(np.multiply(convImg[r_start:r_end,c_start:c_end],kernel))
        
        if (isShow==True):
            cv2.imshow('convolution',convImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return convImg
    else:
        print("請輸入灰階圖片")
        return 0


def convolution(Img,kernel,isShow=True):
    if(Img.ndim==2):
        convImg =  cv2.filter2D(Img, -1, kernel)
    elif(Img.ndim==3):
        convImg = np.copy(Img)
        convImg[:,:,0] = cv2.filter2D(Img[:,:,0], -1, kernel)
        convImg[:,:,1] = cv2.filter2D(Img[:,:,1], -1, kernel)
        convImg[:,:,2] = cv2.filter2D(Img[:,:,2], -1, kernel)
    
    else:
        print('維度不等於2或3')
        convImg = 0
        
    if (isShow==True):
        cv2.imshow('convolutionImg',convImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return convImg

# In[5]  濾波器
# =============================================================================
#  不同濾波器
# =============================================================================

def laplacian(Img,kernelsize,ddepth=-1,isShow=True):
    """ dst = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) """
    ### 
    ### Laplace filter 影像銳化會明顯地加強雜訊
    lapImg = cv2.Laplacian(Img,ddepth,ksize=kernelsize)
    if (isShow==True):
        cv2.imshow('laplacianImg',lapImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return lapImg

def gaussian_blur(Img,kernelsize,sigma,isShow=True):
    """  Img = GaussianBlur(src, (mask_size, mask_size), sigma) """
    ### 高斯模糊化
    ### 該filter2D 會呈現高斯分布，然後再以convolution 方式進行運算，
    ### 2 個參數需要控制，mask 大小及標準差
    ### Sigma 數值越大意味著越遠的的像素會有較大的權值，使得模糊效果更明顯
    ### 低通 高頻雜訊不會通過
    gaussianImg = cv2.GaussianBlur(Img, (kernelsize,kernelsize), sigma)
    
    if (isShow==True):
        cv2.imshow('gaussianImg',gaussianImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return gaussianImg

def bilateral_filter(Img, kernelsize, color_sigma, space_sigma, isShow=True):
    """ Img = cv2.bilateralFilter(src, mask_size, color_sigma, space_sigma) """
    ### 雙邊濾波器
    ### 此種方法的好處為:不但擁有Median filter的除噪效果，
    ### 又能保留圖片中的不同物件的邊緣 (其它三種方式均會造成邊緣同時被模糊化)
    ### 缺點是Bilateral Filter執行的效率較差，運算需要的時間較長
    ### 除傳入的圖片以及Ｋ值大小之外，還需要兩個參數color σ及space σ來計算權重，
    ### 因此Bilateral在計算上除考慮像素之間幾何上的靠近程度之外，
    ### 還多考慮了像素之間的光度及色彩差異，這也為什麼Bilateral Filter會被稱為雙邊的原因
    bilaImg = cv2.bilateralFilter(Img, kernelsize, color_sigma, space_sigma)
    
    if (isShow==True):
        cv2.imshow('bilateralImg',bilaImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bilaImg

def median_blur(Img, kernelsize, isShow=True):
    """ Img = cv2.medianBlur(src, mask_size) """
    ### 中值濾波器
    ### 此種模糊化的方法也經常應用於相片的除噪噪（salt-and-pepper noise）
    ### 也是給予一個KxK 大小的方形window，但是Median是找出所有所有點(最中央那個點除外)的中間值
    ### Median Filter所使用的那個點是個既存的像素而非計算出來的像素，
    ### 因此這是它可以被應用在除雜訊功能上的原因

    medianImg = cv2.medianBlur(Img, kernelsize)
    
    if (isShow==True):
        cv2.imshow("medianImg",medianImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return medianImg


###################
### 邊緣檢測
###################
def sobel(Img,kernelsize ,dx, dy, ddepth=-1, isShow=True):
    """ Img = cv2.Sobel(src, ddepth, dx, dy, ksize) """
    ### ddepth : -1(深度與輸入影像相同)、 CV_8U、CV_16U、CV_16S、CV_32F、CV_64F
    ### dx/dy : 以 x/y 方向求一階導數， 填dx=1,dy=0 或 dx=0,dy=1
    ### Sobel filter 影像銳化會明顯地加強雜訊
    ### Sobel operator 有單一方向性，故會比Laplace 放大雜訊效果還嚴重

    sobelImg = cv2.Sobel(Img,ksize=kernelsize,dx=dx,dy=dy ,ddepth=ddepth)
    
    if (isShow==True):
        cv2.imshow('sobelImg',sobelImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return sobelImg
 
    
def canny(Img,threshold1,threshold2,apeturesize=3,L2=0,isShow=True):
    """ Img = cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]]) """
    ### 邊緣檢測
    ### 其中較大的閾值2用於檢測圖像中明顯的邊緣，但一般情況下檢測的效果不會那麼完美，
    ### 邊緣檢測出來是斷斷續續的。所以這時候用較小的第一個閾值用於將這些間斷的邊緣連接起來。
    ### 可選參數中apertureSize就是Sobel算子的大小
    ### L2gradient參數是一個布林值
    
    cannyImg = cv2.Canny(Img,threshold1,threshold2,apertureSize=apeturesize,L2gradient=L2)
    
    if (isShow==True):
        cv2.imshow('cannyImg',cannyImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cannyImg
    


# In[6]

# =============================================================================
# 霍夫轉換
# =============================================================================

def hough(colorImg,filter_grayImg,threshold,r=1,theta=1,line_len=1000, colorBGR=(0,0,255) ,\
          linewidth=2 ,isShow=True):
    """ lines = cv2.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn]]]): binary image """
    ### 做霍夫前要先將原圖的灰階做邊緣之類的處理唷
    
    ### colorImg 是原圖 彩色版
    ### filter_grayImg 是指做完處理的灰階圖片，可能做了 sobel 或 canny等等
    ### threshold 是指有幾條線通過同一個點就判斷有直線
    ### (r, theta) 是極座標的單位 (長度,角度)
    ### line_len : 是指要畫的線長度
    ### colorBGR : 可以設定線的顏色
    ### linewidth : 可以設定線的粗度
    houghImg = np.copy(colorImg)
    lines = cv2.HoughLines(filter_grayImg ,r ,theta*np.pi/180 ,threshold)  #輸出為極座標的r、theta二維矩陣
    lines = lines[:,0,:]  # 將原本(n,1,2)三維轉成(n,2)二維矩陣
    for r, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)       
        x0 = r*a
        y0 = r*b  
        
        x1 = int(x0 + line_len*(-b)) ## 參數式  x = x0 + a delta(step) #a: x方向向量
        y1 = int(y0 + line_len*(a))  ## #a: x方向向量
        x2 = int(x0 - line_len*(-b)) ## 反向: step = 1000
        y2 = int(y0 - line_len*(a))  ## 反向
        
        cv2.line(houghImg,(x1,y1),(x2,y2),colorBGR,2) ## 圖片要彩色
        
    if (isShow==True):
        cv2.imshow('houghlineImg',houghImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return houghImg

def houghP(colorImg,filter_grayImg,threshold,minLineLength,maxLineGap,\
           r=1,theta=1,colorBGR=(0,255,0),linewidth=2,isShow=True):
    """ Lines = cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) """
    ### 做霍夫前要先將原圖的灰階做邊緣之類的處理唷
    
    ### colorImg 是原圖 彩色版
    ### filter_grayImg 是指做完處理的灰階圖片，可能做了 sobel 或 canny等等
    ### threshold 是指有幾條線通過同一個點就判斷有直線
    ### minLineLength : 線的最短長度，比這個線段短的都忽略掉
    ### maxLineGap :  兩條直線之間的最大間隔，小於此值，就認為是一條直線
    ### (r, theta) 是極座標的單位 (長度,角度)
    ### colorBGR : 可以設定線的顏色
    ### linewidth : 可以設定線的粗度
    
    ### 採取一種概率挑選機制，不是所有的點都進行計算，而是隨機的選取一些點來進行計算，
    ### 這樣的話在閾值設定上也需要降低一些。
    
    houghPImg = np.copy(colorImg)
    lines = cv2.HoughLinesP(filter_grayImg,r,theta* np.pi/180,threshold,minLineLength,maxLineGap)
    lines = lines[:,0,:] # 將原本(n,1,4)三維轉成(n,4)二維矩陣
    for x1,y1,x2,y2 in lines:
        cv2.line(houghPImg,(x1,y1),(x2,y2),colorBGR,linewidth) ## 圖片要彩色
    
    if (isShow==True):
        cv2.imshow('hough(P)_Img',houghPImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return houghPImg




# In[7]
# =============================================================================
# Contour 輪廓   
# =============================================================================
    
def contour(colorImg,binaryImg,color=(0,0,255),\
            mode= cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE,\
            contourIdx=0,thickness=1,isShow=True):
    """ img, countours, hierarchy = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])  \\\\
        cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])"""
    
    ### contour 的物件偵測 物件必須是白色唷
    ### 返回照片 還有 contour list(座標) 兩個值唷!!
    ##############################################################################
    ### cv2.findContours 會返回三個值(3.X版)，分别是img, countours, hierarchy
    
    ###### 第二個參數 mode 表示輪廓的檢索模式，有四種
    ### cv2.RETR_EXTERNAL表示只檢測外輪廓
    ### cv2.RETR_LIST檢測的輪廓不建立等級關係
    ### cv2.RETR_CCOMP建立兩個等級的輪廓，上面的一層為外邊界，裡面的一層為內孔的邊界信息。
    ### 如果內孔內還有一個連通物體，這個物體的邊界也在頂層。
    ### cv2.RETR_TREE建立一個等級樹結構的輪廓。
    ###### 第三個參數method為輪廓的近似辦法
    ### cv2.CHAIN_APPROX_NONE存儲所有的輪廓點，相鄰的兩個點的像素位置差不超過1
    ### cv2.CHAIN_APPROX_SIMPLE壓縮水平方向
    ### cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain近似算法
    ################################################################################
    ### cv2.drawContours 不會返回參數，會直接覆蓋在圖片，所以記得拷貝
    
    ###### 第二個參數contours 是一個list
    ###### 第三個參數 contourIdx，第三個參數指定繪製輪廓list中的哪條輪廓，如果是-1，則繪製其中的所有輪廓
    ###### 第四個參數 thickness，表明輪廓線的寬度，如果是-1（cv2.FILLED），則為填充模式
    
    ## binaryImg 只能輸入二值圖 灰階也不行
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
    ### 這個函數將把原本的contours list用面積做篩選 預設是>1000加入輸出的list
    ### limit填你需要的條件 記得是字串哦!!!
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
    ### 在彩色影像上框出多個邊界
    ### cv2.drawContours 不會返回參數，會直接覆蓋在圖片，所以記得拷貝
    
    ###### 第二個參數contours 是一個list
    ###### 第三個參數 contourIdx，第三個參數指定繪製輪廓list中的哪條輪廓，如果是-1，則繪製其中的所有輪廓
    ###### 第四個參數 thickness，表明輪廓線的寬度，如果是-1（cv2.FILLED），則為填充模式
    
    ## binaryImg 只能輸入二值圖 灰階也不行

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
    ### 先執行這個 得到的feature_list 可以用在下面的函式上
    ###  contours 有兩層哦 [ [], ]
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

# =============================================================================
# 自己寫的contour
# =============================================================================

def polylines(anyImg,contour_np,color=[0,255,0],thickness=3,isClosed=True,isShow=False):
    """ cv2.polylines(image, pts=[points], isClosed=True, color=red_color, thickness=3) """
    ### 輸出是一張標有多邊形邊框的圖片，跟下面的roi不同哦
    ### contours_np 要輸入np.array 的 [[x1,y1],[x2,y2],...]
    ### color 就是邊框的顏色
    ### thickness是邊框粗細    
    ### isClosed是指畫的線終點是否閉合到起點
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
    ### 輸入任何一張照片 及 一個mask(contour) 會產生一張只有那個區域的圖片(輸入彩色輸出就是彩色)
    ###
    ### mask 是一個numpy陣列 可以填多個點 [ [x1,y1],[x2,y2],...,[xn,yn] ]
    ### color 可以設計你感興趣的區域的顏色 填一個0代表黑色
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
    ### 可以一次將含有許多輪廓的地方roi 輸出一張roi後的三通道圖(輸入灰階輸出就會是灰階)
    ### 會使用到上面的函數roi 唷
    roi_out_img = anyimg.copy()
    roi_out_img[:,:,:]= 0
    for ro in contours:
        roiImg = roi(anyimg, ro,isShow=0)
        roi_out_img = roi_out_img|roiImg
        
    return roi_out_img

# In[8]
# =============================================================================
# 膨脹 侵蝕
# =============================================================================

def dilate(img, kernel= None, iterations = 1, isShow=True):
    ### 發現也可以用在彩色的哦!!
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
    ### 發現也可以用在彩色的哦!!
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

def thinning(img, isShow = True):
    if len(img.shape) ==3:
        cvImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        cvImg = img.copy()
    thinned = cv2.ximgproc.thinning(cvImg)
    if isShow:
        # cv2.imshow("inputImg", img)
        # cv2.waitKey(0)
        cv2.imshow("Thinning", thinned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    return thinned


# In[9]
# =============================================================================
# 形態學 Morphology (其實前面侵蝕膨脹也是但分開比較好看)
# =============================================================================
def kernel_3kinds(shape=(3,3),mode=cv2.MORPH_RECT):
    """ kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,15)) """
    ### mode 總共三種
    ###  cv2.MORPH_RECT     mode = 0     : 方形 元素都是 1
    ###  cv2.MORPH_CROSS    mode = 1  : 十字形 1
    ###  cv2.MORPH_ELLIPSE  mode = 2   : 橢圓形 1
    ###  shape 隨你填 輸出是kernel
    return cv2.getStructuringElement(mode, shape)
def morphology(anyImg,mode,kernel,isShow=True):
    """  mophaImg = cv2.morphologyEx(Img, mode, kernel) """
    ###  mode 有幾種:     
    ###  斷開(opening)         : cv2.MORPH_OPEN       mode=2
    ###  閉合(closing)         : cv2.MORPH_CLOSE      mode=3
    ###  梯度(gradient)        : cv2.MORPH_GRADIENT   mode=4
    ###  白頂帽(Top Hat)       : cv2.MORPH_TOPHAT     mode=5
    ###  黑頂帽(Black Top-Hat) : cv2.MORPH_BLACKHAT   mode=6
    ###  H-M Transform         : cv2..MORPH_HITMISS   mode=7   只能用在grayImg
    ###  mode = 0、1  應該分別是 侵蝕 膨脹
    ### kernel 可以用上面的kernel_3kinds
    
    mophaImg = cv2.morphologyEx(anyImg, mode, kernel)
    if (isShow==True):
        cv2.imshow("morphologyImg",mophaImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mophaImg

# In[10]
# =============================================================================
# 幾何轉換 Geometry
# =============================================================================

### 縮放
def resize(anyImg,pixels=None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC,isShow=False):
    """ cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst """
    ### pixels 請填圖片縮放後的像素 例如: (100,100)
    ### 如果沒填pixels 可以填 fx、fy 指X軸、Y軸各縮放幾倍
    
    resizeImg = cv2.resize(anyImg,pixels,fx=fy,fy=fy,interpolation=interpolation)
    if(isShow==True):
        cv2.imshow("resize",resizeImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return resizeImg
        
### 旋轉
def rotate(anyImg,angle,pad_color=(0,0,0),isShow=True):
    """ cv2.getRotationMatrix2D(旋轉中心,角度,旋轉後縮放大小)
         cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst"""
    ### ange 請填上角度 比如轉45度 就填 45
    ###     src - 输入图像。
    ### M - 变换矩阵。
    ### dsize - 输出图像的大小。
    ### flags - 插值方法的组合（int 类型！）
    ### borderMode - 边界像素模式（int 类型！）
    ### borderValue - （重点！）边界填充值; 默认情况下，它为0。
    
    if(anyImg.ndim==3):
        rows,cols,channel = anyImg.shape
    elif(anyImg.ndim==2):
        rows.cols = anyImg.shape
        
    else:
        print("Img is empty!!")
        return None
    
    center = (cols/2, rows/2)   ### 圖片中心點像素
    Matrix = cv2.getRotationMatrix2D(center, angle, 1) ## 得到旋轉矩陣
    rotateImg = cv2.warpAffine(anyImg, Matrix, (rows,cols), borderValue = pad_color)
    
    if isShow==True:
        print("旋轉矩陣 :\n",Matrix)
        cv2.imshow("rotateImg",rotateImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rotateImg

###平移
def translation(img, translate = (5, 3), scale = 1, pad_color = (255, 255, 255), isShow = True): ## translate = (tx, ty)
    height, width = img.shape[:2] 
    ty, tx = height / 2, width / 2
    ## build the matrix
    tx = translate[0]
    ty = translate[1]
    T = np.float32([[1, 0, tx], [0, 1, ty]]) ## translate matrix
    im_t = cv2.warpAffine(img, T, (int(height* scale), int(width* scale)), borderValue=pad_color)
    if isShow:
        cv2.imshow("Tranlated", im_t)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return im_t

###用邊緣像素補充邊緣
def cvPadding(cvImg, top_margin, bottom_margin, left_margin, right_margin, border_type = cv2.BORDER_CONSTANT, pad_color =(255,255,255), isShow = True): ## copy border
    """
    border_type = cv2.BORDER_CONSTANT or BORDER_REPLICATE
    """
    dst = cv2.copyMakeBorder(cvImg, top_margin, bottom_margin, left_margin, right_margin, borderType=border_type, value =pad_color)
    if isShow:
        cv2.imshow("Maker border cvImg", dst)    
        c = cv2.waitKey(-1)
    return dst


### 仿射轉換
def getAffineTransform(original_coor,output_coor):
    """ M = cv2.getAffineTransform(pts1,pts2) """
    ### 仿射轉換矩陣 Affine Transform Matrix
    ### 座標請用 np.float32()
    
    Matrix = cv2.getAffineTransform(original_coor,output_coor)
    return Matrix

def warpAffine(anyImg,Matrix,isShow=True):
    """ dst = cv2.warpPerspective(anyImg,Matrix,dsize) """
    ### 仿射轉換 Affine Transform
    ### 座標請用 np.float32()

    dsize = (anyImg.shape[0],anyImg.shape[1])
    affineImg = cv2.warpAffine(anyImg,Matrix,dsize)
    if(isShow==True):
        plt.subplot(121),plt.imshow(anyImg),plt.title('Input')
        plt.subplot(122),plt.imshow(affineImg),plt.title('Output')
        plt.show()
    return affineImg



### 透視轉換
def getPerspectiveTransform(original_coor,output_coor):
    """ M = cv2.getPerspectiveTransform(pts1,pts2) """
    ### 透視轉換 Perspective Transformation
    Matrix = cv2.getPerspectiveTransform(original_coor,output_coor)
    return Matrix
    
def warpPerspective(anyImg,Matrix,dsize=None,isShow=True):
    """ dst = cv2.warpPerspective(anyImg,Matrix,dsize) """
    if(dsize==None):
        dsize = (anyImg.shape[0],anyImg.shape[1])
    perspectiveImg = cv2.warpPerspective(anyImg,Matrix,dsize)
    
    if(isShow==True):
        plt.subplot(121),plt.imshow(anyImg),plt.title('Input')
        plt.subplot(122),plt.imshow(perspectiveImg),plt.title('Output')
        plt.show()
    return perspectiveImg


# In[11]
# =============================================================================
#  Harris Corner
# =============================================================================
def harris_corner(colorImg,
                  blockSize=2, ksize=3, k=0.04 ,isShow=False):
    """ dst = cv2.cornerHarris(grayImg,blockSize,ksize,k) """
    ### 可以偵測圖片內的corner
    ###
    ### blockSize -角點檢測中要考慮的領域大小
    ### ksize - Sobel求導中使用的窗口大小，感覺類似kernel的存在
    ### k - Harris角點檢測方程中的自由參數,取值參數為[0,04,0.06]

    harrisImg = colorImg.copy()
    gray = cv2.cvtColor(harrisImg,cv2.COLOR_BGR2GRAY)
    ### 一定要換成浮點數32
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,blockSize,ksize,k)  ## dst是一個顯示corner白點的img
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    harrisImg[dst>0.01*dst.max()]=[0,0,255]
    if isShow:
        cv2.imshow('harris_corner_img',harrisImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return harrisImg





# =============================================================================
# Local Feature
# =============================================================================

def orb(grayImg1,grayImg2,match_num=10,isShow=True):
    """ """
    ### match_num 是指 兩張圖片最相關的幾筆特徵 如: 10 就會將兩張圖的最前面10個覺得相關的連起來
    detector = cv2.ORB_create()

    keypoints1, descriptors1 = detector.detectAndCompute(grayImg1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(grayImg2, None)
    
    #-- Step 2: Matching descriptor vectors with a brute force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    
    # Sort matches in the order of their distances
    matches = sorted(matches, key = lambda x : x.distance)
    #-- Draw matches
    img_matches = np.empty((max(grayImg1.shape[0], grayImg2.shape[0]), grayImg1.shape[1]+grayImg2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(grayImg1, keypoints1, grayImg2, keypoints2, matches[:match_num], img_matches)
    
    #-- Show detected matches
    if isShow:
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_matches



# =============================================================================
# 多圖拼接
# =============================================================================

def stitchImgs(anyImglist,isShow=True):
    """ You can use any number of images """
    ### 記得將圖片imread後 放到list 或 tuple 裡
    stitcher = cv2.createStitcher(False)
    status, result = stitcher.stitch(anyImglist)
    
    if isShow:
        cv2.imshow('stitch_image',result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result




# =============================================================================
# 雙鏡頭目標深度估計 Stereo
# =============================================================================

def stereo(grayimg1,grayimg2,numDisparities=16,blockSize=15,isShow=False):
    """ """
    ###numDisparities： 即最大視差值與最小視差值之差,窗口大小必須是16的整數倍，int型
    ### blockSize： 匹配的塊大小。它必須是> = 1的奇數。通常情況下，它應該在3--11的範圍內。這裡設置為大於11也可以，但必須為奇數。
    
    if grayimg1.shape != grayimg2.shape:
        raise Exception('stereo 圖片大小不一樣大!!')
    
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity_img = stereo.compute(grayimg1,grayimg2)
    disparity_img = np.absolute(disparity_img).astype(np.uint8)
    
    if isShow:
        # plt.imshow(disparity,'gray')
        # plt.show()
        cv2.imshow('stereoImg',disparity_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return disparity_img



# In[13]
# =============================================================================
# color map
# =============================================================================


def colormap(anyimg,colormap=cv2.COLORMAP_HOT,isShow=False):
    """ cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) → dst 
        cv2.applyColorMap(grayImg, colormap)"""
    ### colormap 有21種 0~12 可以填數字 也可以使用cv2的參數名稱
    ###常用的是 hot (11)  、 jet (2)
    ### https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
    
    if(anyimg.ndim==3):
        grayImg = cv2.cvtColor(anyimg, cv2.COLOR_BGR2GRAY)
    elif(anyimg.ndim==2):
        grayImg = anyimg
    else:
        raise IndentationError('請輸入灰階或彩色影像')
    
    ### 將數值限縮在0~255之間
    grayImg = cv2.normalize(grayImg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    ### 轉換為colormap
    colormapImg = cv2.applyColorMap(grayImg, colormap=colormap)
    if isShow:
        cv2.imshow('colormap_img',colormapImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return colormapImg

### 調色盤
def palette():
    def nothing(x):
        pass
    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('image')
    
    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)
    
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)
    
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')
    
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]
    
    cv2.destroyAllWindows()

def color_space(colorimg):
    """ """
    ### 這函數可以分割正方體的色彩值
    ### https://blog.csdn.net/weixin_44493841/article/details/102483184
    ### 本來要用 cv2.inRange 但這個的閥值輸出是灰階的樣子 所以改了好久 QQ
    
    try:
        os.makedirs('./outputImg')
    except FileExistsError:
        pass
    
    csImg = colorimg.copy()
    def function(x):
        pass
    
    cv2.namedWindow('ColorSpace')   ### 創建窗口 名字叫 ColorSpace
    
    ### 創建滑動窗口在ColorSpace窗口內
    cv2.createTrackbar('ch1_min', "ColorSpace", 0, 255, function)
    cv2.createTrackbar('ch1_max', "ColorSpace", 0, 255, function)
    cv2.createTrackbar('ch2_min', "ColorSpace", 0, 255, function)
    cv2.createTrackbar('ch2_max', "ColorSpace", 0, 255, function)
    cv2.createTrackbar('ch3_min', "ColorSpace", 0, 255, function)
    cv2.createTrackbar('ch3_max', "ColorSpace", 0, 255, function)
    
    ### 設置滑動窗口初始默認值
    cv2.setTrackbarPos('ch1_min', "ColorSpace",0)
    cv2.setTrackbarPos('ch1_max', "ColorSpace",255)
    cv2.setTrackbarPos('ch2_min', "ColorSpace",0)
    cv2.setTrackbarPos('ch2_max', "ColorSpace",255)
    cv2.setTrackbarPos('ch3_min', "ColorSpace",0)
    cv2.setTrackbarPos('ch3_max', "ColorSpace",255)
    
    while True:
        cv2.imshow('color_space_img',csImg)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        ## 按下 s 或 enter 儲存照片
        elif k== ord('s') or k == 13:  ### 13是enter鍵
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            fn = './outputImg/'+date_time+'color_space.jpg'
            cv2.imwrite(fn,csImg)
            print("儲存成功!!")
        
        
        ### 取得滑動窗口的值
        ch1_min = cv2.getTrackbarPos('ch1_min', "ColorSpace")
        ch1_max = cv2.getTrackbarPos('ch1_max', "ColorSpace")
        ch2_min = cv2.getTrackbarPos('ch2_min', "ColorSpace")
        ch2_max = cv2.getTrackbarPos('ch2_max', "ColorSpace")
        ch3_min = cv2.getTrackbarPos('ch3_min', "ColorSpace")
        ch3_max = cv2.getTrackbarPos('ch3_max', "ColorSpace")
        
        csImg = colorimg.copy()
        
        chmin = np.array([ch1_min,ch2_min,ch3_min])
        chmax = np.array([ch1_max,ch2_max,ch3_max])
        
        channels = cv2.split(csImg)        
        for i in range(3):
            min_bool = channels[i]>=chmin[i]
            max_bool = channels[i]<=chmax[i]
            boo = min_bool&max_bool    ### 範圍內的所有值接True
            channels[i][:]=0
            channels[i][boo] = 0xFF    ### 將True改成255
            csImg[:,:,i] =  csImg[:,:,i]&channels[i]
        
        
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
    
    
    
 # In[14] 
# =============================================================================
# Pattern Match  
# =============================================================================

def template_match(img_color, template, method = 0, isShow = True):
    """ res = cv2.matchTemplate(img, template, method) """
    ### template是要搜尋的照片，灰階
    ### 方法總共6種
    ### res: 存放比對後之分數影像 (儲存相似度)
    t_start = time.perf_counter()
    img_c = img_color.copy()
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # if  not(method in methods):
    #     print("Not valid method")
    #     return
    #print("Matching method", methods[method])
    res = cv2.matchTemplate(img, template, method)
    #print(res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    t_end = time.perf_counter()
    print("Matching Time: ", round((t_end - t_start), 3), " sec.")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  ## min is matched
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_c, top_left, bottom_right, (255, 255, 0), 2)
    if isShow:
        cv2.imshow("Matching Score Matrix:", cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)) #
        cv2.imshow("Match Result:", img_c)
        cv2.waitKey(0)
    return img_c, res


def template_match_multiple(img_color, template, method =0, threshold = 0.8, isShow = True):
    ### template  要使用 np.uint8

    t_start = time.perf_counter()
    img_c = img_color.copy()
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #print("Matching method", methods[method])
    #m = eval(methods[method])
    res = cv2.matchTemplate(img, template, method)
    res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)  ## 正規化
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  ## min is matched
       res = 1 - res
       print(res)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    t_end = time.perf_counter()
    print("Matching Time: ", round((t_end - t_start), 3), " sec.")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    loc = np.where( res >= threshold) ## loc: [[x1, x2, ...] [y1, y2, ....]]
        # =============================================================================
        #     注意!!
        # =============================================================================
    print(len(loc[0]))   #### none max suppresion
    for pt in zip(*loc[::-1]): ## 配對位置 (x, y)
        cv2.rectangle(img_c, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 1)
    if isShow:
        cv2.imshow(methods[method] + "Matching Score Matrix:", res) #
        cv2.imshow(methods[method] + "Match Result:", img_c)
        cv2.waitKey(0)
    return img_c, res

########################################################################################
### 下面的函數主要用在最下面的 template_match_scale 函數
def draw_cross(image, pt, length = 5, color =(0, 0, 255), lineWidth = 1, isShow =True):
    (x, y) = pt
    start_pt = (x-length, y)
    end_pt = (x+length, y)
    image  = cv2.line(image, start_pt, end_pt, color, lineWidth)
    start_pt = (x, y-length)
    end_pt = (x, y + length)
    image  = cv2.line(image, start_pt, end_pt, color, lineWidth)
    if isShow:
        cv2.imshow("draw cross image",image)
        cv2.waitKey(0)
    return image

def find_contour(img, threshold =127, isBlur = True, isShow = True, color = (0, 0, 255)):
    if len(img.shape) == 3:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgray = img.copy()
    if isBlur:
        imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
    ## remember: object is white in a black background
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    findImg,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = color #[(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0), (255, 255, 0),(0, 0, 255), (0, 255, 255), (255, 0, 255),]
    img_contour = img.copy()
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.drawContours(img_contour, [cnt], 0, color, 1)
    #print(contours[0])
    #print(hierarchy)
    if isShow:
        cv2.imshow("Gray image", imgray)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("contours", img_contour)
        cv2.imshow("image", img)
        cv2.waitKey(0)
    return contours, img_contour


def crop_image(image, bbox):
    image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return image_crop

def template_match_scale(img_color, template_color, method =0, threshold = 0.9, minScale = 0.5, maxScale = 1.5,  isShow = True):
    """
    threshold: 相似度
    minScale, maxScale: 將影像放大縮小的範圍
    """
    t_start = time.perf_counter()
    img_c = img_color.copy()
    if len(img_color.shape) == 3:
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    else:
        img = img_color.copy()
    
    if len(template_color.shape) == 3:
        template = cv2.cvtColor(template_color.copy(), cv2.COLOR_BGR2GRAY)
    else:
        template = template_color.copy()
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #print("Matching method", methods[method])
    m = eval(methods[method])
    max = 0  ## max matching score
    max_scale =0
    max_index = 0 ## record the image index
    i = 0
    #res_list = list()
    max_res = None
    ## find the most similar scale
    for scale in np.arange(minScale, maxScale, 0.1):
        temp = cv2.resize(template.copy(), None, fx = scale, fy = scale)
        #print(temp.shape)
        res = cv2.matchTemplate(img, temp, method)

        if m == 5:  ## min is matched
            res = 1 - res
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > max:
            max = max_val
            max_index = i  ## image index
            max_scale = scale
            max_res = res.copy()
            max_res_loc = max_loc
        i = i + 1
        #res_list.append(res)
    #print("Found the most similar scale: ", max_scale, max_index)
    ## find the location in max_index image
    map = np.where(res > threshold, 255, 0)
    loc = np.where( map >= threshold)
    #print(loc)
    map = np.array(map, dtype = np.uint8)
    map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    ##map_thinned = cv2.ximgproc.thinning(map)  ## cannot find the center of the blob
    contours, img_contour = find_contour(map, threshold =127, isBlur = False, isShow = False, color = (0, 0, 255))
    feature_list = calc_contour_feature(contours)
    ## Note find centroid is not good, so try to find the max
    pt_list = list()
    for f in feature_list:
        box = f[3]
        box = [box[0], box[1], box[0]+box[2], box[1]+ box[3]]
        res_roi = crop_image(max_res, box)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_roi)
        pt_list.append((max_loc[0] + box[0], max_loc[1]+box[1]))
    #print(pt_list)
    h = int(h* max_scale)
    w = int(w * max_scale)
    #print(h," : ", w)
    t_end = time.perf_counter()
    print("Matching Time: ", round((t_end - t_start), 3), " sec.")
    max_res = cv2.cvtColor(np.array(max_res*255, np.uint8), cv2.COLOR_GRAY2BGR)
    max_res_disp = max_res.copy()
    for pt in pt_list: #zip(*loc[::-1]): ## 配對位置 (x, y)
        cv2.rectangle(img_c, pt, (int(pt[0]) + w, int(pt[1]) + h), (0, 0, 255), 1)
        max_res_disp = draw_cross(max_res_disp, pt, isShow=False)
    if isShow:
        cv2.imshow(methods[m] + "Matching Score Matrix:", max_res_disp) #
        cv2.imshow(methods[m] + "Match Result:", img_c)
        cv2.waitKey(0)
    return img_c, max_res  

#%%

def draw_bbox2(img, bbox2, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox2 = img.copy()
    box = np.int0(cv2.boxPoints (bbox2))  #–> int0會省略小數點後方的數字
    img_bbox = cv2.drawContours(img_bbox2, [box], -1, color, width)
    if isShow:
        cv2.imshow("image with bbox", img_bbox2)
        cv2.waitKey(0)
    return img_bbox2

def shape_match(image, template, threshold, method = 1):
    ## input image must be grayscale
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ret, thresh = cv2.threshold(template, 127, 255, 0)
    ret, thresh2 = cv2.threshold(image, 127, 255, 0)
    findImg,contours, hierarchy= cv2.findContours(thresh, 2, 1)  # 所有內外contours
    cnt_base = contours[0]
    findImg,contours, hierarchy = cv2.findContours(thresh2, 2, 1)
    image_disp = image_color.copy()
    cv2.drawContours(image_disp, contours, -1, (0, 0, 255), 1)
    print("Found contours: ", len(contours))
    cv2.imshow("All contours", image_disp)
    cv2.waitKey(0)
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
                (255, 255, 0)]
    contour_list = list()
    score_list = list()
    for i, cont in enumerate(contours):
        similiarity = cv2.matchShapes(cnt_base, cont, method, 0.0) ## method I, 0.0 none
        print(similiarity)
        if similiarity < threshold: ## smaller is better
            M = cv2.moments(cont)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(image_color, [cont], 0, color[i%5], 1)
            cv2.putText(image_color, str(round(similiarity, 4)), (cX - 20, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color[i%5], 1)
            contour_list.append(cont)
            score_list.append(similiarity)
    return contour_list, score_list, image_color

def shape_match2(image, template, aspectThreshold = 0.1, keepAspectRatio = True, 
            isBlack=True, method = 1, minNumContour =  10, isShow = True):
    ## input image must be grayscale
    ## template
    ## minNumContour: 有時候會有一些雜訊，需要過濾掉不的contour
    ## 預設: object 的aspect ratio 會保持不變，所以 ratio 差必須小於 0.1
    if len(image.shape)>2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if len(template.shape)>2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    ret, thresh = cv2.threshold(template, 127, 255, isBlack)
    findImg,contours, hierarchy = cv2.findContours(thresh^0xFF, 2, 1)  # 所有內外contours
    cv2.drawContours(template_color, contours, -1, (0, 0, 255), 1)
    #cv2.waitKey(0)
    template_cnt = contours[0]
    (x,y),(w,h), angle = cv2.minAreaRect(template_cnt)
    if w > h:
        temp = w
        w = h
        h = temp
    template_aspect_ratio = float(w)/h
    #print("Template: ", w, h, template_aspect_ratio)
    ## detecting image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ret, thresh2 = cv2.threshold(image, 127, 255, 0)
    findImg,contours, hierarchy = cv2.findContours(thresh2, 2, 1)
    image_disp = image_color.copy()
    cv2.drawContours(image_disp, contours, -1, (0, 0, 255), 1)
    print("Found contours: ", len(contours))
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
                (255, 255, 0)]
    contour_list = list()
    score_list = list()
    for i, cont in enumerate(contours):
        if len(cont) < minNumContour:  ## ignor small object
            continue
        (x, y), (w, h), angle = rect = cv2.minAreaRect(cont)
        box = np.int0(cv2.boxPoints(rect))
        if w > h:
            temp = w
            w = h
            h = temp
        aspect_ratio = float(w)/h
        #print(aspect_ratio, template_aspect_ratio)
        
        if abs(aspect_ratio - template_aspect_ratio)>0.3:
            continue
        similiarity = cv2.matchShapes(template_cnt, cont, method, 0.0) ## method I, 0.0 none
        #print(similiarity)
        if similiarity < aspectThreshold: ## smaller is better
            print(w, h, aspect_ratio)
            M = cv2.moments(cont)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(image_color, [box], 0, color[i%5], 1)
            cv2.putText(image_color, str(round(similiarity, 4)), (cX - 20, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color[i%5], 1)
            contour_list.append(cont)
            score_list.append(similiarity)
    if isShow:
        cv2.imshow("Template contour:", template_color)
        cv2.imshow("All contours", image_disp)
        cv2.imshow("Matched contours", image_color)
        cv2.waitKey(0)
    return contour_list, score_list, image_color

# In[15]
# =============================================================================
#  傅立葉
# =============================================================================

def fft(grayImg):
    f = np.fft.fft2(grayImg)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(f))
    s2 = np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original')
    plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center')

    return s2





#%%   main

def main():
    # img = cv2.imread('lena.jpg',0)
    
    # histImg = calcAndDrawHist(img, [0,0,255])
    # eqImg = histogram_equalization(img)
    
    # threshImg = threshold(img,150,255,allmethodShow=True)
    
    # thresh = average_intensity_threshold(img)     #輸出不是照片唷 是平均閥值
    # thresh = modified_iterative_method(img)
    # threshImg = threshold(img,thresh,255,allmethodShow=True)
    # otsuImg = otsu(img)
    # adImg = adaptive_threshold(img, 11, 2)
    
    
    
    
    # convImg1 = convolution(img,filters('sobel_x'))
    # convImg = myconvolution(img,filters('averaging'))
    # convImg = convolution(img,filters('sobel_vertical'))
    # convImg = convolution(img,filters('sobel_vertical2'))
    # convImg = convolution(img,filters('vertical2'))
    # convImg = convolution(img,filters('horizon'))
    # convImg = convolution(img,filters('horizon2'))
    
    
    # medianImg = median_blur(img,7)
    # gauImg = gaussian_blur(img,3,0)
    # bilaImg = bilateral_filter(img,15,75,75)
    # sobelImg = sobel(img,3 ,1,0)
    # cannyImg = canny(img,50,150)
    # lapImg = laplacian(img,3)
          
    # mask_size  = 5
    # kernel = np.ones((mask_size, mask_size), dtype = np.float32) /25
    # meanImg = cv2.filter2D(img, -1, kernel)    
    
    
    
    
    ### 霍夫轉換
    # img1 = cv2.imread('sudoku.jpg',1)
    # img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # img2 = canny(img1, 50, 150)
    
    # houghImg = hough(img1, img2, 200)
    # houghPImg = houghP(img1,img2,100,20,20)
    # cv2.imshow('lena',img2)
    # cv2.waitKey(0)
    
    # cv2.imshow('anotherlena',cannyImg)
    # cv2.waitKey(0)
    
    # cv2.imshow('lenahist',equ)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

if __name__=='__main__':
    main()