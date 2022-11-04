#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用

using namespace std;

void add_L(cv::Mat& src, cv::Mat& l);
void add_R(cv::Mat& src, cv::Mat& l);
//######################################################################################################
int main() {
	//創建資料夾
	char folderName[] = "bonus_img";

	if (_access(folderName, 0) == -1)
	{
		_mkdir(folderName);
	}


	//圖片路徑
	const char* path_L = "../WM5fb22625ebf39.bmp";
	const char* path_R = "../WM5fb2262892679.bmp";
	cv::Mat src_img_L = cv::imread(path_L);
	cv::Mat src_img_R = cv::imread(path_R);
	
	//創建 480x900 的圖片
	cv::Mat stitchImg = cv::Mat::zeros(480, 900, CV_8UC3);

	//放入左圖
	add_L(stitchImg, src_img_L);
	cv::imwrite("./bonus_img/1_only_leftImg.bmp", stitchImg);

	
	//開始咯 好累
	cv::Mat dst_warp, dst_warpRotateScale, dst_warpTransformation, dst_warpFlip;
	cv::Point2f origP[4];//原圖中的四點 ,一個包含三維點（x，y）的數組，其中x、y是浮點型數
	cv::Point2f cvtP[4];//目標圖中的四點  

	//原始座標點
	origP[0] = cv::Point2f(0, 0);
	origP[1] = cv::Point2f(0, src_img_R.rows);
	origP[2] = cv::Point2f(src_img_R.cols, 0);
	origP[3] = cv::Point2f(src_img_R.cols, src_img_R.rows);

	//轉換後座標點
	cvtP[0] = cv::Point2f(250,130);
	cvtP[1] = cv::Point2f(310,470);
	cvtP[2] = cv::Point2f(770, 0);
	cvtP[3] = cv::Point2f(900,475);

	cv::Mat new_R = cv::Mat::zeros(480, 900, CV_8UC3);
	
	

	add_R(new_R, src_img_R);

	cv::Mat Matri = cv::getPerspectiveTransform(origP, cvtP);
	warpPerspective(new_R, new_R, Matri, stitchImg.size());//仿射轉換
	cv::imwrite("./bonus_img/2_onlyRightImg.bmp", new_R);

	for (int r = 0; r< new_R.rows; r++) {
		for (int c = 0; c < new_R.cols; c++) {
			if (new_R.at<cv::Vec3b>(r, c)[0] > 0|| new_R.at<cv::Vec3b>(r, c)[1] > 0) {
				stitchImg.at<cv::Vec3b>(r, c)[0] = new_R.at<cv::Vec3b>(r, c)[0];
				stitchImg.at<cv::Vec3b>(r, c)[1] = new_R.at<cv::Vec3b>(r, c)[1];
				stitchImg.at<cv::Vec3b>(r, c)[2] = new_R.at<cv::Vec3b>(r, c)[2];
			}
		}
	}
	
	cv::imwrite("./bonus_img/3_finish.bmp", stitchImg);
	cv::imshow("Test", stitchImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

//######################################################################################################
void add_R(cv::Mat& src, cv::Mat& l) {
	for (int r = 0; r < l.rows; r++) {
		for (int c = 0; c < l.cols; c++) {
			for (int i = 0; i < 3; i++) {
				src.at<cv::Vec3b>(r, c)[i] = l.at<cv::Vec3b>(r, c)[i];
			}
		}
	}
}
void add_L(cv::Mat& src, cv::Mat& l) {
	for (int r = 80; r <= 449; r++) {
		for (int c = 0; c <= 509; c++) {
			for (int i = 0; i < 3; i++) {
				src.at<cv::Vec3b>(r, c)[i] = l.at<cv::Vec3b>(r-80, c)[i];
			}
		}
	}
}
