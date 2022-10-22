import cv2
import numpy as np
import os
from yuyi_DIP_tool import *
import copy
import json

# img_list, imgName_list = imreadImgs('./all_img')

img_data={'img_name' : None,
  'cell_num' : None,
  'cells_area' : [],
  'center' : [],
  'nucleus_num' :[],
  'nucleus_total_area' : [],
  'ratio' : [],
  'deal' : None,
  'cell_parameter' : [],
  'nuc_parameter' : []}





def find_cell(img,color_choose,s_min,s_max,ker,morph_mode,erod_times,dilat_times,decision_min,decision_max):
    colorimg = img.copy()
    ker = (1+ker)*2+1
    if(color_choose==3 or color_choose==4 or color_choose==5):    
        colorimg = cv2.cvtColor(colorimg, cv2.COLOR_BGR2HSV)


    low = np.array([s_min,s_min,s_min])
    upper = np.array([s_max,s_max,s_max])
    colorimg = color_space_set_lower_upper(colorimg,low,upper,isShow=0)
    c1,c2,c3 = bgr(colorimg)
    if(color_choose==0 or color_choose==3):
        c = c1
    if(color_choose==1 or color_choose==4):
        c = c2
    if(color_choose==2 or color_choose==5):
        c = c3
        
    c = otsu(c,isShow=0)
    kernel = kernel_3kinds((ker,ker),2)
    c = erode(c,kernel,iterations=erod_times,isShow=0)
    c = dilate(c,kernel,iterations=dilat_times,isShow=0)
    if morph_mode==0:
        c = c
    elif morph_mode==1:
        c = c^0xFF
    out,con = contour(img, c,color=(0,255,255),isShow=0)
    
    mask_min = '>'+str(decision_min)
    mask_max = "<"+str(decision_max)
    maskR = contour_area_choose(con,limit = mask_min)
    maskR = contour_area_choose(maskR,limit = mask_max)
    
    
    out_img = poly_contour(img, maskR,color=(0,255,255),isShow=0) 
    return [out_img,c,maskR]

def cell_func(img,l):
    out = find_cell(img,l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8])
    return out

def imgshow(img_list):
    cv2.imshow('output',img_list[0])
    cv2.imshow('otsu',img_list[1])



    
def use(img_name,parameter_img_name, path = './all_img/',parameter_path = './json/',save_json=False):
    with open(parameter_path+parameter_img_name+'.json','r') as Obj:     
        total_json = json.load(Obj)
    cell_parameter_list = total_json["cell_parameter"]
    nuc_parameter_list = total_json["nuc_parameter"]

    # img_name = '002.jpg'
    ## 讀單張
    inputImg = cv2.imread(path+img_name,1)
    size = (800,600)
    inputImg = cv2.resize(inputImg, size)
    
    img = inputImg.copy()
    cell_data = copy.deepcopy(img_data)
    draw_cell_list = []
    draw_nuc_list = []
    
    
    
    celcon = []
    for i in cell_parameter_list:
        o = cell_func(img,i)
        celcon.append(copy.deepcopy(o[2]))
    
    ### 將contours合併
    cell_contours = []
    if(len(cell_parameter_list)==1):
        cell_contours = celcon[0]
    else:
        for i in range(len(cell_parameter_list)):
            for j in celcon[i]:
                cell_contours.append(j)



    #### 輸出 cell_contours
    ###-------------------------------------------------------------
    for cell_contour in cell_contours:
        roiImg = roi(img,cell_contour)
        nuccon = []
        for i in nuc_parameter_list:
            o = cell_func(roiImg,i)
            nuccon.append(o[2])
        
        ### 將contours合併
        nuc_contours = []
        if(len(nuc_parameter_list)==1):
            nuc_contours = nuccon[0]
        else:
            for i in range(len(nuc_parameter_list)):
                for j in nuccon[i]:
                    nuc_contours.append(j)
        #### 輸出 nuc_contours
        
        
        
        
        
        ### 找不到細胞核就不用了
        if(len(nuc_contours)==0):
            # print("找不到細胞核")
            continue
        else:
            nucleus_num = len(nuc_contours)
            nucleus_total_area = 0
            for nuc_contour in nuc_contours:
                nucleus_total_area += cv2.contourArea(nuc_contour)  ###計算全部細胞核面積
                ### 畫圖用list
                draw_nuc_list.append(nuc_contour)
        cells_area = cv2.contourArea(cell_contour)   ### 計算細胞面積
        
        ### 計算質心
        M = cv2.moments(cell_contour)
        if cells_area != 0: ## same as M["m00"] !=0
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        
        ### 核質比
        ratio = round(nucleus_total_area/cells_area,3)
        if(ratio>1):
            nucleus_total_area = nucleus_total_area-cells_area
        ratio = round(nucleus_total_area/cells_area,3)
            
        ### 畫核質比
        words(img, str(ratio), (cx,cy),color=(131,252,252),bold=2,scale=0.5)
    
        
        ### 畫圖用list
        draw_cell_list.append(cell_contour)
        
        
        
        
        
        
        ### save
        if save_json:
            cell_data["cells_area"].append(cells_area)
            cell_data["center"].append(center)
            cell_data["nucleus_num"].append(nucleus_num)
            cell_data["nucleus_total_area"].append(nucleus_total_area)
            cell_data["ratio"].append(ratio)
    if save_json:
        cell_data["img_name"] = img_name
        cell_data["cell_num"] = len(draw_cell_list)
        cell_data["cell_parameter"] = cell_parameter_list
        cell_data["nuc_parameter"] = nuc_parameter_list
        cell_data["deal"] = True
        ### 儲存成json
        save_file = './json/'+img_name+'.json'
        json.dump(cell_data,open(save_file,"w"),indent=2)   
        
    
    
    
    ##畫圖
    text = "cell_num :"+str(len(draw_cell_list))
    words(img, text, (10,40),color=(0,0,0),bold=2,scale=0.9)
    outImg = poly_contour(img, draw_cell_list,color=(0,255,255),isShow=0) 
    outImg = poly_contour(outImg, draw_nuc_list,color=(255,255,0),isShow=1)
    ### 儲存圖片
    result_file = "./outputImg/"
    try:
        if os.path.isfile(result_file+img_name):
            os.remove(result_file+img_name)
        cv2.imwrite(result_file+img_name,outImg)
    except:
        print("儲存圖片失敗")

def create_files():
    try:
        os.makedirs('./json')
    except FileExistsError:
        pass
    try:
        os.makedirs('./result')
    except FileExistsError:
        pass    
 
def main():
    create_files()
    path = './test_img/'       ### 改成圖片放置的檔案路徑
    
    #### method 1 可以一次推論多張圖片
    # im = ["001 -1.bmp","001.bmp","004.bmp","005.bmp","013.bmp","014.bmp","021.bmp","032.bmp","043.bmp","055.bmp"]
    # for na in im:
    #     img_name = na
    #     parameter_img = "001 -1.bmp"
    #     use(img_name,parameter_img,path)
    
    
    #### method 2 一次推論一張照片
    img_name = "055.jpg"              ### 輸入要推論的圖片名稱
    parameter_img = "001 -1.jpg"      ### 剛剛用了set.py的那張圖片名稱，使用他的參數推論其他圖片
    use(img_name,parameter_img,path)
    
if __name__=="__main__":
    main()