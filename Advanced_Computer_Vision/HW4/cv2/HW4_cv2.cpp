#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用

using namespace std;
using namespace cv;

Mat template_matching(Mat& pre, Mat& now ,Mat& colorImg, Mat* output_grayImg, Mat* output_difImg);
int main() {
	// 建立資料夾
	char folderName_gray[] = "gray_images";
	if (_access(folderName_gray, 0) == -1)
	{
		_mkdir(folderName_gray);
	}

	char folderName_color[] = "color_images";
	if (_access(folderName_color, 0) == -1)
	{
		_mkdir(folderName_color);
	}

	char folderName_differ[] = "differ_images";
	if (_access(folderName_differ, 0) == -1)
	{
		_mkdir(folderName_differ);
	}

	char folderName_mp4[] = "output_mp4";
	if (_access(folderName_mp4, 0) == -1)
	{
		_mkdir(folderName_mp4);
	}

	//宣告變數
	Mat frame;
	Mat colorImg, grayImg, preImg, nowImg;
	Mat outputImg;
	Mat output_grayImg;
	Mat output_difImg;

	int count = 0;
	char text[80];

	const char* src = "../WM61b1c81394b01.avi";   // 修改輸入影片路徑
	VideoCapture capture;

	frame = capture.open(src);
	if (!capture.isOpened())
	{
		cout << "無法開啟影片，請確認路徑!!" << endl;
		exit(-1);
	}
	namedWindow("video", WINDOW_AUTOSIZE);


	//影片儲存初始化
	Size s = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
	int fps = capture.get(CAP_PROP_FPS);
	VideoWriter writer("./output_mp4/color.mp4", CAP_OPENCV_MJPEG, fps, s, true);
	VideoWriter writer_gray("./output_mp4/gray.mp4", CAP_OPENCV_MJPEG, fps, s, true);

	//影片串流
	while (capture.read(frame)) {
		cvtColor(frame, grayImg, COLOR_BGR2GRAY);
		
		grayImg.copyTo(nowImg);
		//影像處理
		if (count>0) {
			
			// 影像處理
			outputImg = template_matching(preImg, nowImg, colorImg, &output_grayImg, &output_difImg);

			//輸出影片
			imshow("video", outputImg);
			waitKey(10);
			

			//儲存影片
			writer.write(outputImg);
			writer_gray.write(output_grayImg);

			//儲存照片
			sprintf_s(text, "./gray_images/%d_%d.bmp", count, count + 1);
			imwrite(text, output_grayImg);
			sprintf_s(text, "./differ_images/%d.bmp",count + 1);
			imwrite(text, output_difImg);
			sprintf_s(text, "./color_images/%d_%d.bmp", count,count+1);
			imwrite(text, outputImg);

		}
		
		
		count++;
		nowImg.copyTo(preImg);
		frame.copyTo(colorImg); // 彩色的 t-1影像
	}
	capture.release();
	writer.release();
	return 0;
}


void find_exhaustive_range(int r, int c, int size_r, int size_c, Rect* box, int* original_x, int* original_y) {
	if (c == 0) {
		if (r == 0) {
			box->width = 60;
			box->height = 60;
			*original_x = 0;  //模板左上角座標在這張圖片的所在位置
			*original_y = 0;  //模板左上角座標在這張圖片的所在位置
		}
		else if (r == size_r - 1) {
			box->width = 60;
			box->height = 60;
			*original_x = 0; //模板左上角座標在這張圖片的所在位置
			*original_y = 20; //模板左上角座標在這張圖片的所在位置
		}
		else {
			box->width = 60;
			box->height = 80;
			*original_x = 0; //模板左上角座標在這張圖片的所在位置
			*original_y = 20; //模板左上角座標在這張圖片的所在位置
		}
	}
	else if (c == size_c - 1) {
		if (r == 0) {
			box->width = 60;
			box->height = 60;
			*original_x = 20; //模板左上角座標在這張圖片的所在位置
			*original_y = 0; //模板左上角座標在這張圖片的所在位置
		}
		else if (r == size_r - 1) {
			box->width = 60;
			box->height = 60;
			*original_x = 20; //模板左上角座標在這張圖片的所在位置
			*original_y = 20; //模板左上角座標在這張圖片的所在位置
		}
		else {
			box->width = 60;
			box->height = 80;
			*original_x = 20; //模板左上角座標在這張圖片的所在位置
			*original_y = 20; //模板左上角座標在這張圖片的所在位置
		}
	}
	else {
		if (r == 0) {
			box->width = 80;
			box->height = 60;
			*original_x = 20; //模板左上角座標在這張圖片的所在位置
			*original_y = 0; //模板左上角座標在這張圖片的所在位置
		}
		else if (r == size_r - 1) {
			box->width = 80;
			box->height = 60;
			*original_x = 20; //模板左上角座標在這張圖片的所在位置
			*original_y = 20; //模板左上角座標在這張圖片的所在位置
		}
		else {
			box->width = 80;
			box->height = 80;
			*original_x = 20; //模板左上角座標在這張圖片的所在位置
			*original_y = 20; //模板左上角座標在這張圖片的所在位置
		}
	}
}




Mat template_matching(Mat& pre, Mat& now, Mat& colorImg, Mat* output_grayImg, Mat* output_difImg) {
	int rows = pre.rows;
	int cols = pre.cols;
	Mat out_img = Mat::zeros(rows, cols, CV_8UC3);
	Mat out_gray_img = Mat::zeros(rows, cols, CV_8UC1);
	Mat out_dif_img = Mat::zeros(rows, cols, CV_8UC1);
	Mat exhaustive_img;
	Mat temp_img;
	Mat match_img;
	Mat cut;

	int grid_r = rows / 40;
	int grid_c = cols / 40;
	Rect coord_temp(0, 0, 40, 40);
	Rect coord_exhau(0, 0, 80, 80);
	char text[80];
	int pre_x, pre_y;
	int ex_x, ex_y;
	int vector_x, vector_y;
	int aft_x, aft_y;
	double minVal, maxVal;
	Point minLoc, maxLoc;

	for (int c = 0; c < grid_c; c++) {
		for (int r = 0; r < grid_r; r++) {
			pre_x = c * 40;
			pre_y = r * 40;

			//找exhaustive 
			coord_exhau.x = ((pre_x - 20) < 0) ? (0) : (pre_x - 20);
			coord_exhau.y = ((pre_y - 20) < 0) ? (0) : (pre_y - 20);
			find_exhaustive_range(r, c, grid_r, grid_c, &coord_exhau, &ex_x, &ex_y);
			cut = Mat(now, coord_exhau);
			cut.copyTo(exhaustive_img);

			//找 template
			coord_temp.x = pre_x;
			coord_temp.y = pre_y;
			cut = Mat(pre, coord_temp);
			cut.copyTo(temp_img);

			//模板匹配
			matchTemplate(exhaustive_img, temp_img, match_img, TM_CCOEFF_NORMED);
			minMaxLoc(match_img, &minVal, &maxVal, &minLoc, &maxLoc);
			
			vector_x = (int)maxLoc.x - ex_x;
			vector_y = (int)maxLoc.y - ex_y;
			
			//移動
			aft_x = pre_x + vector_x;
			aft_y = pre_y + vector_y;

			//cout << pre_x << "," << pre_y<<endl;
			for (int r = 0; r < 40;r++) {
				for (int c = 0; c < 40; c++) {
					for (int i = 0; i < 3; i++) {
						// 移動grid到新的位置  ( 彩色 )
						out_img.at<Vec3b>(aft_y + r, aft_x + c)[i] = colorImg.at<Vec3b>(pre_y + r, pre_x + c)[i];
					}
					// 移動grid到新的位置  ( 灰色 )
					out_gray_img.at<uchar>(aft_y + r, aft_x + c) = pre.at<uchar>(pre_y + r, pre_x + c);
				}
			}


			
		}
	}
	
	//算 now 與 predict 差值
	out_dif_img = now ^ out_gray_img;

	// return
	*output_difImg = out_dif_img;
	*output_grayImg = out_gray_img;
	return out_img;
}