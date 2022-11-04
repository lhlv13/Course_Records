#include<stdio.h>
#include<stdlib.h>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用
#include<math.h>

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
	Mat* img;
	short** direction; //儲存角度值
}sobel_direct;



Mat imread(const char* path);
void imwrite(const char* path, Mat img);
Mat copyImg(Mat&);
Mat color2Gray(Mat*);
Mat binarizing(Mat*, uch, char);
Mat dilate(Mat* src, const char direction, int iteration);
Mat erode(Mat* src, const char direction, int iteration);

Mat birdEyeView(Mat&);
Mat crop(const Mat& src, int x1, int y1, int x2, int y2);
void find_range(Mat& src, int* x1, int* y1, int* x2, int* y2);
void word_to_img(const Mat& src, int* u, int* v, const double x, const double y);

Mat sobel(Mat& src, sobel_direct& sobel_direct);
Mat find_sobel_angle_range(sobel_direct& src_direct, const short a, const short b);
void mix_img(Mat& output_img, Mat& src, Mat& mask, const char color);
Mat merge(Mat& up, Mat& down);
Mat repair(Mat& src, Mat& cpy);
//######################################################################################################
int main() {

	//創建資料夾
	char folderName[] = "img_by_C";

	if (_access(folderName, 0) == -1)
	{
		_mkdir(folderName);
	}

	const char* path = "../WM5fb2261fb1a7b.bmp";
	Mat inputImg;
	inputImg = imread(path);

	// crop 出需要的部分
	Mat cropImg;
	double alpha = 15 * 3.14159 / 180;
	double calc = (inputImg.rows - 1) * (-0.025 + alpha) / (2 * alpha);
	int uHorizon = (int)calc;
	printf("uHorizon: %d\n", uHorizon);
	cropImg = crop(inputImg, uHorizon, 0, 767, 1023);
	imwrite("./img_by_C/1_crop_img.bmp", cropImg);

	//轉成鳥瞰圖
	Mat birdImg;
	birdImg = birdEyeView(cropImg);
	imwrite("./img_by_C/2_bird.bmp", birdImg);
	//-------------------------------
	//     畫車道線
	//-------------------------------
	// 灰階
	Mat grayImg;
	grayImg = color2Gray(&cropImg);
	imwrite("./img_by_C/3_gray.bmp", grayImg);

	////二值化
	//Mat binaryImg;
	//binaryImg = binarizing(&grayImg, 200, 'f');
	//imwrite("./img_by_C/4_test_binary.bmp", binaryImg);

	//邊緣偵測
	Mat sobelImg;
	sobel_direct sobelImg_direct;
	sobelImg = sobel(grayImg, sobelImg_direct);
	imwrite("./img_by_C/4_sobel.bmp", sobelImg);

	//////////
	//找線
	/////////

	//最左邊的線
	Mat L_angleImg;
	L_angleImg = find_sobel_angle_range(sobelImg_direct, 91, 130);
	imwrite("./img_by_C/5_L_angle.bmp", L_angleImg);

	Mat L_morImg;
	L_morImg = erode(&L_angleImg, 'h', 2);
	L_morImg = dilate(&L_morImg, 'h', 7);
	L_morImg = dilate(&L_morImg, 'v', 7);
	imwrite("./img_by_C/6_L_morpha.bmp", L_morImg);

	//中間線
	Mat M_angleImg;
	M_angleImg = find_sobel_angle_range(sobelImg_direct, 320, 340);
	imwrite("./img_by_C/7_M_angle.bmp", M_angleImg);

	Mat M_morImg;

	M_morImg = dilate(&M_angleImg, 'v', 10);
	M_morImg = dilate(&M_morImg, 'h', 10);
	imwrite("./img_by_C/8_M_morpha.bmp", M_morImg);

	//右邊線
	Mat R_angleImg;
	R_angleImg = find_sobel_angle_range(sobelImg_direct, 10, 50);
	imwrite("./img_by_C/9_R_angle.bmp", R_angleImg);

	Mat R_morImg;
	R_morImg = erode(&R_angleImg, 'v', 1);
	R_morImg = erode(&R_morImg, 'h', 1);
	R_morImg = dilate(&R_morImg, 'h', 7);
	R_morImg = dilate(&R_morImg, 'v', 7);
	
	imwrite("./img_by_C/10_R_morpha.bmp", R_morImg);

	//mix
	Mat mixImg;
	mix_img(mixImg, cropImg, L_morImg, 'b');
	mix_img(mixImg, mixImg, M_morImg, 'r');
	mix_img(mixImg, mixImg, R_morImg, 'g');
    imwrite("./img_by_C/11_mix.bmp", mixImg);

	//融合圖片
	Mat mergeImg;
	mergeImg = merge(inputImg, mixImg);
	imwrite("./img_by_C/12_merge.bmp", mergeImg);

	//轉換鳥瞰圖
	Mat threeImg;
	threeImg = birdEyeView(mixImg);
	imwrite("./img_by_C/13_b.bmp", threeImg);


	//修補線
	Mat repairImg;
	repairImg = repair(threeImg, birdImg);
	imwrite("./img_by_C/14_repair.bmp", repairImg);
}
//######################################################################################################
Mat repair(Mat& src,Mat& cpy) {
	Mat tmp = copyImg(cpy);
	int b, g, red;
	int* lines_rows = (int*)calloc(3,sizeof(int));
	for (int i = 0; i < 3; i++) {
		lines_rows[i] = -1;
	}
	for (int c = 0; c < src.cols; c++){
		for (int r = 0; r < src.rows; r++) {
			b = src.image[0][r][c];
			g = src.image[1][r][c];
			red = src.image[2][r][c];

			if(b==255 && g==0 &&red==0){
				lines_rows[0] = r;
			}
			else if (b == 0 && g == 255 && red == 0) {
				lines_rows[1] = r;
			}
			else if (b == 0 && g == 0 && red == 255) {
				lines_rows[2] = r;
			}
		}
		if (lines_rows[0] != -1 && lines_rows[1] != -1 && lines_rows[2] != -1) { break; }
	}
	printf("repair: (%d,%d,%d)\n", lines_rows[0], lines_rows[1], lines_rows[2]);
	//lines_rows[1] = (lines_rows[0] + lines_rows[2]) / 2;
	int ro;
	for (int i = 0; i < 3; i++){
		ro = lines_rows[i];
		for(int c = 0; c < src.cols; c++) {
			tmp.image[0][ro][c] = 0;
			tmp.image[1][ro][c] = 0;
			tmp.image[2][ro][c] = 0;

			if (i == 0) {
				tmp.image[0][ro][c] = 255;
			}
			if (i == 1) {
				tmp.image[1][ro][c] = 255;
			}
			if (i == 2) {
				tmp.image[2][ro][c] = 255;
			}
		}
	}

	Mat output_img;
	output_img = crop(tmp, 184, 0, 250, 66);
	return output_img;
}


Mat merge(Mat& up, Mat& down) {
	Mat output_img = up;
	
	//記憶體配置
	output_img.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++) {
		output_img.image[i] = (uch**)calloc(up.rows, sizeof(uch*));
		for (int r = 0; r < up.cols; r++) {
			output_img.image[i][r] = (uch*)calloc(up.cols, sizeof(uch));
		}
	}
	
	int r_change, c_change;
	r_change = up.rows - down.rows;

	for (int r = 0; r < up.rows; r++) {
		for (int c = 0; c < up.cols; c++) {
			if (r < r_change) {
				output_img.image[0][r][c] = up.image[0][r][c];
				output_img.image[1][r][c] = up.image[1][r][c];
				output_img.image[2][r][c] = up.image[2][r][c];
			}
			else {
				output_img.image[0][r][c] = down.image[0][r - r_change][c];
				output_img.image[1][r][c] = down.image[1][r - r_change][c];
				output_img.image[2][r][c] = down.image[2][r - r_change][c];
			}
		}
	}

	return output_img;
}





void mix_img(Mat& output_img, Mat& src, Mat& mask, const char color) {
	if (&output_img.image != &src.image) {
		output_img = copyImg(src);
	}
	

	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (mask.image[0][r][c]>0) {
				output_img.image[0][r][c] = 0;
				output_img.image[1][r][c] = 0;
				output_img.image[2][r][c] = 0;

				if (color == 'r') {
					output_img.image[2][r][c] = 255;
				}
				else if (color == 'g') {
					output_img.image[1][r][c] = 255;
				}
				else {
					output_img.image[0][r][c] = 255;
				}
			}
		}
	}

}

Mat find_sobel_angle_range(sobel_direct& src_direct,const short a,const short b) {
	Mat output_img;
	//記憶體配置
	output_img.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++) {
		output_img.image[i] = (uch**)calloc(src_direct.img->rows, sizeof(uch*));
		for (int r = 0; r < src_direct.img->rows; r++) {
			output_img.image[i][r] = (uch*)calloc(src_direct.img->cols, sizeof(uch));
		}
	}
	
	short small, large, value;
	small = (a > b) ? b : a;
	large = (a > b) ? a : b;
	
	for (int r = 0; r < src_direct.img->rows; r++) {
		for (int c = 0; c < src_direct.img->cols; c++) {
			value = src_direct.direction[r][c];
			if (value >= small && value <= large) {
				for (int i = 0; i < 3; i++) {
					output_img.image[i][r][c] = 255;
				}
			}
		}
	}

	

	return output_img;
}


Mat sobel(Mat& src, sobel_direct& sobel_direct) {
	//傳入的圖片是灰階圖，只需要對一個通道運算

	Mat sobel_img = src;
	// img記憶體配置
	sobel_img.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++) {
		sobel_img.image[i] = (uch**)calloc(src.rows, sizeof(uch*));
		for (int r = 0; r < src.rows; r++) {
			sobel_img.image[i][r] = (uch*)calloc(src.cols, sizeof(uch));
		}
	}

	// 梯度方向的記憶體配置
	sobel_direct.direction = (short**)calloc(sobel_img.rows, sizeof(short*));
	for (int r = 0; r < sobel_img.rows; r++) {
		sobel_direct.direction[r] = (short*)calloc(sobel_img.cols, sizeof(short));
	}

	//filter X
	double sobel_x[3][3] = { {-1, 0, 1},
							 {-2, 0, 2},
							 {-1, 0, 1 } };

	// filter Y
	double sobel_y[3][3] = { { 1, 2, 1},
							 { 0, 0, 0},
		                     { -1,-2,-1 }};
	
	//test
	double maxv=45;
	double minv=45;


	// 運算
	double sum_x, sum_y;
	double magnitude;
	double direction;
	for (int r = 1; r < sobel_img.rows-1; r++) {
		for (int c = 1; c < sobel_img.cols-1; c++) {
			//捲積
			sum_x = 0.0;
			sum_y = 0.0;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					sum_x += (double)src.image[0][r + i][c + j] * sobel_x[i+1][j+1];
					sum_y += (double)src.image[0][r + i][c + j] * sobel_y[i+1][j+1];
				}
			}
			sum_x /= 8;
			sum_y /= 8;
			sum_x = (sum_x > 255) ? 255 : sum_x;
			sum_y = (sum_y > 255) ? 255 : sum_y;
			// 設值 (大小)
			
			magnitude = sqrt(sum_x * sum_x + sum_y * sum_y);
			magnitude = magnitude * 255 / 360;
			magnitude = (magnitude > 255) ? 255 : magnitude;
			sobel_img.image[0][r][c] = (uch)magnitude;
			sobel_img.image[1][r][c] = (uch)magnitude;
			sobel_img.image[2][r][c] = (uch)magnitude;

			//設值 (方向)
			if (sum_x == 0 && sum_y == 0) {
				direction = -1;
				sobel_direct.direction[r][c] = (short)(direction);   //沒有值得就設定 -1
			}
			else {
				direction = atan2(sum_y, sum_x);          //與正向x軸的角度
				direction = direction * 180.0 / 3.14159; //改成角度
				direction = (direction >=0) ? direction : (360 + direction);  //使角度位於0<= angle <360度
				sobel_direct.direction[r][c] = (short)(direction);
			}
			

			
			// test 輸出cmd用
			maxv = (maxv < direction) ? direction : maxv;
			minv = (minv > direction) ? direction : minv;
		
			


		}
	}
	// test 輸出cmd用
	printf("測試角度(%lf, %lf)\n", maxv, minv);

	sobel_direct.img = &sobel_img;
	return sobel_img;
}




//------------------------------------------------------



Mat crop(const Mat& src, int x1, int y1, int x2, int y2) {
	int height = (x2 - x1 + 1);
	int width = (y2 - y1 + 1);

	//改檔頭
	Mat output = src;
	output.bmpImgHeader.biWidth = output.cols = width;
	output.bmpImgHeader.biHeight = output.rows = height;
	output.alignment = ((size_t)output.cols * (size_t)output.channels * 3) % 4;
	output.size = (size_t)output.cols * (size_t)output.rows;

	//記憶體配置
	output.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++) {
		output.image[i] = (uch**)calloc(height, sizeof(uch*));
		for (int r = 0; r < height; r++) {
			output.image[i][r] = (uch*)calloc(width, sizeof(uch));
		}
	}

	for (int r = x1; r <= x2; r++) {
		for (int c = y1; c <= y2; c++) {
			for (int i = 0; i < 3; i++) {
				output.image[i][r-x1][c-y1] = src.image[i][r][c];
			}
		}
	}
	return output;
}

void find_range(Mat& src, int* x1, int* y1, int* x2, int* y2) {
	*x1 = 0; *y1 = 0; *x2 = 0; *y2 = 0;
	double calculate1;
	double x, y;

	double alpha = 15 * 3.14159 / 180;
	double dx = 4.0;
	double dy = -10.0;
	double dz = 5.0;
	double y0 = 0.0;
	double th0 = 0.25;
	double m = (double)src.rows;
	double n = (double)src.cols;
	
	for (int u = 0; u < m; u++) {
		for (int v = 0; v < n; v++) {
			calculate1 = 1 / tan(th0 - alpha + (2 * alpha * u) / (m - 1));
			calculate1 *= dz*sin(y0 - alpha + (2*alpha*v)/(n-1));

			x = calculate1 + dx;
			y = calculate1 + dy;
			
			*x1 = (*x1 > x) ? ((int)x) : (int)(*x1);   //左上座標 x
			*y1 = (*y1 > y) ? ((int)y) : (int)(*y1);   //左上座標 y
			*x2 = (*x2 < x) ? ((int)x) : (int)(*x2);   //右下座標 x
			*y2 = (*y2 < y) ? ((int)y) : (int)(*y2);   //右下座標 y
		}
	}
	
}




void word_to_img(const Mat& src, int* u, int* v, const double x, const double y){
	double calculate;
	
	double alpha = 15 * 3.14159 / 180;
	double dx = 4.0;
	double dy = -10.0;
	double dz = 5.0;
	double y0 = 0.0;
	double th0 = 0.25;
	double m = (double)src.rows;
	double n = (double)src.cols;

	//calculate = (atan(dz * sin(atan((x - dx) / (y - dy))) / (x - dx)) - th0 + alpha)*(m-1)/(2*alpha);
	calculate = (x - dx) / (y - dy);
	calculate = atan(calculate);
	calculate = dz * sin(calculate) / (x - dx);
	calculate = atan(calculate) - th0 + alpha;
	calculate = calculate * (m - 1) / (2 * alpha);

	*u = (int)calculate;

	//calculate = (atan((x - dx) / (y - dy)) - y0 + alpha)* (n - 1) / (2 * alpha);
	calculate = 0;
	calculate = (x - dx) / (y - dy);
	calculate = atan(calculate) - y0 + alpha;
	calculate = calculate * (n - 1) / (2 * alpha);

	*v = (int)calculate;
}

Mat birdEyeView(Mat& src) {
	int x1, y1, x2, y2;
	// 找變換後圖片的大小 image to word
	find_range(src, &x1, &y1, &x2, &y2);
	printf("TEST(%d,%d,%d,%d)\n", x1, y1, x2, y2);

	Mat bird_img = src;
	int bird_height =(x2-x1)/10;
	int bird_width = (y2-y1)/10;
	printf("bird(%d, %d)\n", bird_height, bird_width);
	
	//更改標頭檔
	bird_img.bmpImgHeader.biWidth = bird_img.cols = bird_width;
	bird_img.bmpImgHeader.biHeight = bird_img.rows = bird_height;
	bird_img.alignment = ((size_t)bird_img.cols * (size_t)bird_img.channels * 3) % 4;
	bird_img.size = (size_t)bird_img.cols * (size_t)bird_img.rows;
	//記憶體配置
	bird_img.image = (uch***)calloc(3, sizeof(uch**));
	for (int i = 0; i < 3; i++) {
		bird_img.image[i] = (uch**)calloc(bird_height, sizeof(uch*));
		for (int r = 0; r < bird_height; r++) {
			bird_img.image[i][r] = (uch*)calloc(bird_width, sizeof(uch));
		}
	}

	//bird轉換
	int u, v;
	double scale = 10;
	double X, Y;
	for (int x = 0; x < bird_img.rows; x++) {
		for (int y = 0; y < bird_img.cols; y++) {
			//找 原圖的轉換座標		
			X = (x * 10.0 + x1) / scale;   //scale讓取的範圍在小一點 
			Y = y * 10.0 / scale;
			word_to_img(src, &u, &v, X, Y);

			if (u >= 0 && v >= 0) {
				if (u < src.rows && v < src.cols) {
					for (int i = 0; i < 3; i++) {
						bird_img.image[i][x][y] = src.image[i][u][v];
					}
				}
			}
		}
	}

	return bird_img;
}






// free(image*)
//void del(Mat& src) {
//	for (int i = 0; i < 3; i++) {
//		for (int r = 0; r < src.rows; r++) {
//			free(src.image[i][r]);
//		}
//		free(src.image[i]);
//	}
//	free(src.image);
//}

//------------------------------------------------------------------------------
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
						if ((i) < (temp.rows - 1))
							dila.image[0][i + 1][j] = 255;
					}
					else {
						if ((j) < (temp.cols - 1))
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
				out.image[0][i][j] = out.image[0][i][j] ^ 0xff;
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
	uch value;
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			value = src->image[2][i][j];
			if (value >= 191) {
				out.image[0][i][j] = value;
				out.image[1][i][j] = out.image[0][i][j];
				out.image[2][i][j] = out.image[0][i][j];
			}
			else {
				out.image[0][i][j] = 0;
				out.image[1][i][j] = 0;
				out.image[2][i][j] = 0;
			}

			/*out.image[0][i][j] = src->image[0][i][j] * 0.114 + src->image[0][i][j] * 0.587 + src->image[0][i][j] * 0.299;
			out.image[1][i][j] = out.image[0][i][j];
			out.image[2][i][j] = out.image[0][i][j];*/
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

