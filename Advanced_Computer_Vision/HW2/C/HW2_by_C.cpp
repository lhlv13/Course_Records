#include<stdio.h>
#include<stdlib.h>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用
#include<math.h>
#include<time.h>  //計算時間
#define PI 3.1415926

typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned char uch;

#pragma pack(2)    //記憶體對齊
typedef struct {    //14 Bytes
	uint16_t bfType;
	uint32_t bfSize;
	uint16_t bfReserved1;
	uint16_t bfReserved2;
	uint32_t bfOffBits;
}FileHeader;

typedef struct {     //40 Bytes
	uint32_t biSize;
	uint32_t biWidth;
	uint32_t biHeight;
	uint16_t biPlanes;
	uint16_t biBitCount;
	uint32_t biCompression;
	uint32_t biSizeImage;
	uint32_t biXPelsPerMeter;
	uint32_t biYPelsPerMeter;
	uint32_t biClrUsed;
	uint32_t biClrImportant;
}ImgHeader;
#pragma pack()     //取消記憶體對齊

#pragma pack(1)
typedef struct {
	FileHeader bmpFileHeader;
	ImgHeader bmpImgHeader;
	uint32_t cols;
	uint32_t rows;
	uint16_t channels;
	size_t size;
	size_t alignment;
	uch*** image;
}Mat;
#pragma pack()

typedef struct {
	int* area;           // 面積
	double** centroids;  // 質心
	int** box;           // box的左上、右下座標
	int* label_list;     // 每隻手連通元件給予的標籤值
	int num;             // 總共判別出幾隻手
	int* longest;        // 最長邊的長度
	int* angle;          // 角度
	int* finger_num;     // 每隻手的手指數量
}Connected;

//計算時間
typedef struct {
	double binarizing;
	double morphology;
	double connectedComponent;
	double propertyAnalysis;
	double drawing;
}Time;

Time mytime;
clock_t begin, end;



Mat imread(const char* path);
void imwrite(const char* path, Mat img);
Mat copyImg(Mat&);
Mat color2Gray(Mat*);
Mat binarizing(Mat *, uch, char);
Mat connetedComponent(Mat *, Connected *);
void collision_func(Mat* ptr, int r, int c, int collision_keep, int collision_del);
void compute_box_area_centroids(Mat* , int , Connected* );
Mat drawBox(Mat*, Connected*, char*);
Mat dilate(Mat* src, const char direction, int iteration);
Mat erode(Mat* src, const char direction, int iteration);
Mat img_minus_img(Mat* src1, Mat* src2);
void cal_hands_fingers(Connected& hands, Connected& fingers);
void show(Connected*);
void show_fingers(Connected* reco);
void show_time(Time*);
//######################################################################################################
int main() {

	//創建資料夾
	char folderName[] = "Image_no_cv2";

	if (_access(folderName, 0) == -1)
	{
		_mkdir(folderName);
	}

	Mat inputImg;

	char* path = "./hand.bmp";
	// 讀取檔案
	inputImg = imread(path);
	
	// Color to Gray
	Mat grayImg;
	grayImg = color2Gray(&inputImg);
	imwrite("./Image_no_cv2/1_gray_no_cv2.bmp", grayImg);

	// Binary Image
	begin = clock();
	Mat binaryImg;
	binaryImg = binarizing(&grayImg, 230, 't');  //'t' 是指 黑白轉換  不轉換請輸入任意字元
	end = clock();
	mytime.binarizing = (double)(end - begin) / CLOCKS_PER_SEC;
	imwrite("./Image_no_cv2/2_binary_no_cv2.bmp", binaryImg);


	//morphology
	begin = clock();
	Mat morphoImg;
	int iter = 5;
	
	morphoImg = erode(&binaryImg, 'v', iter);   // 垂直方向
	morphoImg = erode(&morphoImg, 'h', iter);   // 水平方向
	morphoImg = dilate(&morphoImg, 'v', iter);
	morphoImg = dilate(&morphoImg, 'h', iter);
	end = clock();
	mytime.morphology = (double)(end - begin) / CLOCKS_PER_SEC;
	imwrite("./Image_no_cv2/3_morpha_img_no_cv2.bmp", morphoImg);


	// connected component
	Mat concomImg;
	Connected concom_property;
	concomImg = connetedComponent(&binaryImg, &concom_property);
	imwrite("./Image_no_cv2/4_connect_no_cv2.bmp", concomImg);


	//draw box
	begin = clock();
	Mat boxImg;
	char color[] = {0, 0,255 };
	boxImg = drawBox(&binaryImg, &concom_property, color);
	end = clock();
	mytime.drawing = (double)(end - begin) / CLOCKS_PER_SEC;
	imwrite("./Image_no_cv2/5_box_no_cv2.bmp", boxImg);

	show_time(&mytime);
	//-----------------------------------------------
	//  Bonus
	//-----------------------------------------------
	Mat no_fingerImg;
	iter = 25;
	no_fingerImg = erode(&binaryImg, 'v', iter);
	no_fingerImg = erode(&no_fingerImg, 'h', iter);
	no_fingerImg = dilate(&no_fingerImg, 'v', iter);
	no_fingerImg = dilate(&no_fingerImg, 'h', iter);
	imwrite("./Image_no_cv2/6_Bonus_noFingers_no_cv2.bmp", no_fingerImg);

	Mat fingerImg;
	fingerImg = img_minus_img(&binaryImg, &no_fingerImg);
	iter = 10;
	fingerImg = erode(&fingerImg, 'v', iter);
	fingerImg = erode(&fingerImg, 'h', iter);
	fingerImg = dilate(&fingerImg, 'v', iter);
	fingerImg = dilate(&fingerImg, 'h', iter);
	fingerImg = erode(&fingerImg, 'v', 13);
	fingerImg = dilate(&fingerImg, 'v', 13);

	imwrite("./Image_no_cv2/7_Bonus_Fingers_no_cv2.bmp", fingerImg);


	Mat concom_fingerImg;
	Connected finger_property;
	concom_fingerImg = connetedComponent(&fingerImg, &finger_property);

	Mat box_fingerImg;
	char color_finger[] = { 255, 0, 0 };
	box_fingerImg = drawBox(&boxImg, &finger_property, color_finger);
	imwrite("./Image_no_cv2/8_Bonus_Box_no_cv2.bmp", box_fingerImg);
	
	
	

	//判斷手指是誰的
	cal_hands_fingers(concom_property, finger_property);
	

	//顯示結果
	show(&concom_property);
	
	// 手指的 性質
	printf("\n\n手指的性質");
	show_fingers(&finger_property);
}
//######################################################################################################


//顯示時間
void show_time(Time* t) {
	printf("\n時間 :\n");
	printf("binarizing : %lf 秒\n",t->binarizing);
	printf("morphology : %lf 秒\n",t->morphology);
	printf("connectedComponent : %lf 秒\n",t->connectedComponent);
	printf("propertyAnalysis : %lf 秒\n",t->propertyAnalysis);
	printf("drawing : %lf 秒\n",t->drawing);

}



//顯示結果
void show_fingers(Connected* reco) {
	for (int i = 0; i < reco->num; i++) {
		printf("\n手指%d:\n\t", i + 1);
		printf("area : %d,\t", reco->area[i]);
		printf("centroid : (%f, %f),\n", reco->centroids[i][0], reco->centroids[i][1]);
		printf("\tlong : %d,\t", reco->longest[i]);
		printf("與水平軸的 angle : %d 度\n", reco->angle[i]);
		printf("\tbox座標 : 左上(%d, %d), 右下(%d, %d)", reco->box[i][0], reco->box[i][1], reco->box[i][2], reco->box[i][3]);
	}
	printf("\n共%d隻手指\n\n", reco->num);
}


void show(Connected* reco) {
	for (int i = 0; i < reco->num; i++) {
		printf("\n手掌%d:   手指數 : %d\n\t", i + 1,reco->finger_num[i]);
		printf("area : %d,\t", reco->area[i]);
		printf("centroid : (%f, %f),\n", reco->centroids[i][0], reco->centroids[i][1]);
		printf("\tlong : %d,\t", reco->longest[i]);
		printf("與水平軸的 angle : %d 度\n", reco->angle[i]);
		printf("\tbox座標 : 左上(%d, %d), 右下(%d, %d)", reco->box[i][0], reco->box[i][1], reco->box[i][2], reco->box[i][3]);
	}
	printf("\n共%d隻手掌\n\n", reco->num);
}

//計算每隻手掌有幾隻手指
void cal_hands_fingers(Connected& hands, Connected& fingers) {  
	hands.finger_num = (int*)calloc(hands.num, sizeof(int));


	for (int i = 0; i < hands.num; i++) {
		for (int j = 0; j < fingers.num; j++) {
			if (fingers.centroids[j][0] > hands.box[i][0] && fingers.centroids[j][0] < hands.box[i][2]) {
				if (fingers.centroids[j][1] > hands.box[i][1] && fingers.centroids[j][1] < hands.box[i][3]) {
					hands.finger_num[i]++;   //手指數量 + 1
				}
			}
		}
	}
}


//照片減照片
Mat img_minus_img(Mat* src1, Mat* src2) {
	Mat minus_img = copyImg(*src1);
	int pixel;
	for (int i = 0; i < src1->rows; i++) {
		for (int j = 0; j < src1->cols; j++)
		{
			pixel = (int)src1->image[0][i][j] - (int)src2->image[0][i][j];
			pixel = (pixel > 0) ? (pixel) : (-pixel);
			minus_img.image[0][i][j] = pixel;
			minus_img.image[1][i][j] = pixel;
			minus_img.image[2][i][j] = pixel;

		}
	}
	return minus_img;
}



//morphology
Mat erode(Mat* src, const char direction, int iteration) {
	Mat erod = copyImg(*src);
	Mat temp = copyImg(*src);

	for (int iter = 0; iter < iteration; iter++) {

		for (int i = 0; i < temp.rows; i++) {
			for (int j = 0; j < temp.cols; j++) {
				erod.image[0][i][j] = 0;
				if (temp.image[0][i][j] > 0) {
					if (direction == 'v') {
						if (i < (temp.rows - 1)) {
							if (temp.image[0][i + 1][j] > 0) {
								erod.image[0][i][j] = 255;
							}
						}
					}
					else {
						if (j < (temp.cols - 1)) {
							if (temp.image[0][i][j + 1] > 0) {
								erod.image[0][i][j] = 255;
							}
						}

					}
				}
				temp.image[0][i][j] = erod.image[0][i][j];
			}
		}
	}

	for (int i = 0; i < erod.rows; i++) {
		for (int j = 0; j < erod.cols; j++) {
			erod.image[1][i][j] = erod.image[0][i][j];
			erod.image[2][i][j] = erod.image[0][i][j];
		}
	}
	return erod;
}


Mat dilate(Mat* src, const char direction, int iteration) {
	Mat dila = copyImg(*src);
	Mat temp = copyImg(*src);
	
	for (int iter = 0; iter < iteration; iter++) {

		for (int i = 0; i < temp.rows; i++) {
			for (int j = 0; j < temp.cols; j++) {
				if (temp.image[0][i][j] > 0) {
					if (direction == 'v') {
						if((i)<(temp.rows - 1))
						dila.image[0][i + 1][j] = 255;
					}
					else{
						if((j)<(temp.cols-1))
						dila.image[0][i][j + 1] = 255;
					}
				}
				temp.image[0][i][j] = dila.image[0][i][j];
			}
		}
	}

	for (int i = 0; i < dila.rows; i++) {
		for (int j = 0; j < dila.cols; j++) {
			dila.image[1][i][j] = dila.image[0][i][j];
			dila.image[2][i][j] = dila.image[0][i][j];
		}
	}

	//釋放記憶體
	//free(&temp);
	return dila;
}



//draw box
Mat drawBox(Mat *src, Connected* reco, char* colors) {
	Mat draw_img = copyImg(*src);
	int x1, y1, x2, y2;
	

	for (int n = 0; n < reco->num; n++) {
		x1 = reco->box[n][0];
		y1 = reco->box[n][1];
		x2 = reco->box[n][2];
		y2 = reco->box[n][3];
		for (int i = x1; i <= x2; i++) {
			draw_img.image[0][i][y1] = colors[0];
			draw_img.image[1][i][y1] = colors[1];
			draw_img.image[2][i][y1] = colors[2];
			draw_img.image[0][i][y2] = colors[0];
			draw_img.image[1][i][y2] = colors[1];
			draw_img.image[2][i][y2] = colors[2];

		}
		for (int i = y1; i <= y2; i++) {
			draw_img.image[0][x1][i] = colors[0];
			draw_img.image[1][x1][i] = colors[1];
			draw_img.image[2][x1][i] = colors[2];
			draw_img.image[0][x2][i] = colors[0];
			draw_img.image[1][x2][i] = colors[1];
			draw_img.image[2][x2][i] = colors[2];

		}
	}
	return draw_img;
}



// 連通元件使用的計算函數
void compute_box_area_centroids(Mat* con, int label_num, Connected* reco) {
	reco->label_list = (int*)calloc(label_num, sizeof(int));
	int index = 0;
	int x1, y1, x2, y2;
	
	for (int i = 0; i < con->rows; i++) {
		for (int j = 0; j < con->cols; j++) {

			if (con->image[0][i][j] != 0) {
				// 掃描
				for (int k = 0; k < label_num; k++) {
					//舊標籤
					if (con->image[0][i][j] == reco->label_list[k]) {

						// 賦值給x1,y1,x2,y2
						reco->box[k][0] = (reco->box[k][0] > i) ? i : reco->box[k][0];
						reco->box[k][1] = (reco->box[k][1] > j) ? j : reco->box[k][1];
						reco->box[k][2] = (reco->box[k][2] < i) ? i : reco->box[k][2];
						reco->box[k][3] = (reco->box[k][3] < j) ? j : reco->box[k][3];

						//賦值給 area
						reco->area[k]++;

						//賦值給 centroids
						reco->centroids[k][0] = (reco->centroids[k][0]*(reco->area[k]-1)+i)/ reco->area[k];
						reco->centroids[k][1] = (reco->centroids[k][1] * (reco->area[k]- 1) + j) / reco->area[k];


						break;

					}

					//新標籤
					if (k == (label_num - 1)) {
						reco->label_list[index] = (int)con->image[0][i][j];
						//初始化 (x1,y1,x2,y2)
						reco->box[index][0] = reco->box[index][2] = i;
						reco->box[index][1] = reco->box[index][3] = j;
						//初始化 面積
						reco->area[index] = 1;
						// 初始化 centroids
						reco->centroids[index][0] = (double)i;
						reco->centroids[index][1] = j;

						index++;
					}
				}
			}
		}
	}

	//計算最長邊 和 角度 使用artan(h/w)
	int w, h;
	for (int i = 0; i < label_num; i++) {
		h = reco->box[i][2] - reco->box[i][0];
		w = reco->box[i][3] - reco->box[i][1];
		reco->longest[i] = (h > w) ? h : w;
		reco->angle[i] = atan((double)h/(double)w)*180.0/PI;
	}
	
}

//回朔法
void collision_func(Mat *ptr,int r, int c, int collision_keep, int collision_del) {
	ptr->image[0][r][c] = collision_keep;
	if (ptr->image[0][r - 1][c] == collision_del) {
		collision_func(ptr, r - 1, c, collision_keep, collision_del);
	}
	if (ptr->image[0][r][c - 1] == collision_del) {
		collision_func(ptr, r, c-1, collision_keep, collision_del);
	}
	if (ptr->image[0][r][c+1] == collision_del) {
		collision_func(ptr, r, c+1, collision_keep, collision_del);
	}
}



//連通元件
Mat connetedComponent(Mat* src, Connected* record){
	Mat connect = *src;
	int label = 0, collision=0; 
	int collision_keep, collision_del;
	


	// 產生 零矩陣
	connect.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++)
	{
		connect.image[i] = (uch**)calloc(src->rows, sizeof(uch*));
		for (int j = 0; j < src->rows; j++)
		{
			connect.image[i][j] = (uch*)calloc(src->cols, sizeof(char));
		}
	}




	// 四連通  
	begin = clock();
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			if (src->image[0][i][j] == 0) {   //src.image[0][i][j] 為 0 不做事
				continue;
			}
			else{     //src.image[0][i][j] 有值	
				if (i == 0 && j == 0) {  //第一個元素
					connect.image[0][i][j] = ++label; 
				}
				else if (i == 0 && j != 0) {       //第一列的元素		
					connect.image[0][i][j] = (connect.image[0][i][j-1] == 0) ? (++label) : connect.image[0][i][j - 1];
				}
				else if(i != 0 && j ==0) {         //第一行的元素
					connect.image[0][i][j] = (connect.image[0][i-1][j] == 0) ? (++label) : connect.image[0][i-1][j];
				}
				else {                   //i-1 j-1都不會小於 0 的元素



					if (connect.image[0][i - 1][j] == 0 && connect.image[0][i][j - 1] == 0) { //上左都沒有label
						connect.image[0][i][j] = ++label;
					}
					else if (connect.image[0][i - 1][j] > 0 && connect.image[0][i][j - 1] == 0) {   //上面元素有label
						connect.image[0][i][j] = connect.image[0][i - 1][j];
					}
					else if (connect.image[0][i - 1][j] == 0 && connect.image[0][i][j - 1] > 0) {   //左邊元素有label
						connect.image[0][i][j] = connect.image[0][i][j - 1];
					}


					
					else if (connect.image[0][i - 1][j] > 0 && connect.image[0][i][j - 1] > 0) {
						if (connect.image[0][i - 1][j] == connect.image[0][i][j - 1]) {
							connect.image[0][i][j] = connect.image[0][i - 1][j];
						}
						else {
							collision++;
							//解決衝突
							collision_keep = (connect.image[0][i - 1][j] < connect.image[0][i][j - 1]) ? (connect.image[0][i - 1][j]) : (connect.image[0][i][j - 1]);
							collision_del = (connect.image[0][i - 1][j] > connect.image[0][i][j - 1]) ? (connect.image[0][i - 1][j]) : (connect.image[0][i][j - 1]);
							collision_func(&connect, i, j, collision_keep, collision_del);
						}
					}
				}
			}
		}
	}

	end = clock();
	mytime.connectedComponent = (double)(end - begin) / CLOCKS_PER_SEC;

	
	// 計算 box座標  area centroids  num longest angle
	begin = clock();
	int label_num;   
	label_num = label - collision;
	
	Connected records;
	records.num = label_num;   // bounding_box數量
	records.area = (int*)calloc(label_num, sizeof(int));
	records.longest = (int*)calloc(label_num, sizeof(int));
	records.angle = (int*)calloc(label_num, sizeof(int));
	

	records.centroids = (double**)calloc(label_num, sizeof(double*));
	for (int i = 0; i < label_num; i++) {
		records.centroids[i] = (double*)calloc(2,sizeof(double));  // [x,y]
	}

	records.box = (int**)calloc(label_num, sizeof(int*));
	for (int i = 0; i < label_num; i++) {
		records.box[i] = (int*)calloc(4,sizeof(int));       //  [x1,y1,x2,y2] 左上座標 右下座標
	}

	compute_box_area_centroids(&connect, label_num, &records);

	end = clock();
	mytime.propertyAnalysis = (double)(end - begin) / CLOCKS_PER_SEC;
	


	// 複製給 其他顏色
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			//connect.image[0][i][j] = connect.image[0][i][j]*10;
			connect.image[1][i][j] = connect.image[0][i][j];
			connect.image[2][i][j] = connect.image[0][i][j];

		}
	}


	*record = records;
	return connect;
}


// 二值化
Mat binarizing(Mat* src, uch thres_value, char inverse) {
	Mat out = copyImg(*src);
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			if (src->image[0][i][j] > thres_value)
				out.image[0][i][j] = 255;
			else
				out.image[0][i][j] = 0;

			if (inverse == 't') {
				out.image[0][i][j] = out.image[0][i][j]^0xff;
			}
			else {
				out.image[0][i][j] = out.image[0][i][j];
			}
			
			out.image[1][i][j] = out.image[0][i][j];
			out.image[2][i][j] = out.image[0][i][j];
		}
	}
	return out;
}



// 灰階
Mat color2Gray(Mat* src) {
	Mat out = copyImg(*src);
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			out.image[0][i][j] = src->image[0][i][j] * 0.114+ src->image[0][i][j]*0.587+ src->image[0][i][j]*0.299;
			out.image[1][i][j] = out.image[0][i][j];
			out.image[2][i][j] = out.image[0][i][j];
		}
	}
	return out;
}

// 拷貝
Mat copyImg(Mat& src) {
	Mat out = src;
	out.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++)
	{
		out.image[i] = (uch**)calloc(src.rows, sizeof(uch*));
		for (int j = 0; j < src.rows; j++)
		{
			out.image[i][j] = (uch*)calloc(src.cols, sizeof(char));
		}
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			out.image[0][i][j] = src.image[0][i][j];
			out.image[1][i][j] = src.image[1][i][j];
			out.image[2][i][j] = src.image[2][i][j];
		}
	}
	return out;
}

//---------------------------------------------------------------------
Mat imread(const char* path) {
	FILE* file_ptr;
	fopen_s(&file_ptr, path, "rb+");
	if (!file_ptr) {
		printf("path is error!!");
		exit(1);
	}

	Mat img;
	fread((char*)&img.bmpFileHeader, sizeof(char), sizeof(img.bmpFileHeader), file_ptr);
	fread((char*)&img.bmpImgHeader, sizeof(char), sizeof(img.bmpImgHeader), file_ptr);


	img.cols = img.bmpImgHeader.biWidth;
	img.rows = img.bmpImgHeader.biHeight;
	img.channels = img.bmpImgHeader.biBitCount / 8;
	img.size = (size_t)img.cols * (size_t)img.rows;
	img.alignment = ((size_t)img.cols * (size_t)img.channels * 3) % 4;
	//printf("alignment :  %d", img.alignment);
	
	// 建立三維矩陣
	img.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++)
	{
		img.image[i] = (uch**)calloc(img.rows, sizeof(uch*));
		for (int j = 0; j < img.rows; j++)
		{
			img.image[i][j] = (uch*)calloc(img.cols, sizeof(char));
		}
	}


	fseek(file_ptr, img.bmpFileHeader.bfOffBits, SEEK_SET);

	for (int i = img.rows - 1; i >= 0; i--) {
		for (int j = 0; j < img.cols; j++) {
			fread((char*)&img.image[0][i][j], sizeof(char), sizeof(uch), file_ptr);
			fread((char*)&img.image[1][i][j], sizeof(char), sizeof(uch), file_ptr);
			fread((char*)&img.image[2][i][j], sizeof(char), sizeof(uch), file_ptr);
		}
		fseek(file_ptr, (long)img.alignment, SEEK_CUR);
	}
	fclose(file_ptr);
	return img;
}

void imwrite(const char* path, Mat img) {
	FILE* file_ptr;
	fopen_s(&file_ptr, path, "wb+");
	if (!file_ptr) {
		printf("path is error!!");
		exit(-1);
	}

	fwrite((char*)&img.bmpFileHeader, sizeof(char), sizeof(img.bmpFileHeader), file_ptr);
	fwrite((char*)&img.bmpImgHeader, sizeof(char), sizeof(img.bmpImgHeader), file_ptr);


	for (int i = img.rows - 1; i >= 0; i--) {
		for (int j = 0; j < img.cols; j++) {
			fwrite((char*)&img.image[0][i][j], sizeof(char), sizeof(uch), file_ptr);
			fwrite((char*)&img.image[1][i][j], sizeof(char), sizeof(uch), file_ptr);
			fwrite((char*)&img.image[2][i][j], sizeof(char), sizeof(uch), file_ptr);
		}
		for (size_t i = 0; i < img.alignment; i++)
		{
			fwrite("", sizeof(char), sizeof(uch), file_ptr);
			
		}
	}
	fclose(file_ptr);
}