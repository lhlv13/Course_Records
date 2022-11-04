#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用
#include<ctime>

using namespace cv;
using namespace std;


//計算時間
typedef struct {
	double binarizing;
	double morphology;
	double connectedComponent;
	double propertyAnalysis;
	double drawing;
}Time;


//顯示時間
void show_time(Time* t) {
	printf("\n時間 :\n");
	printf("binarizing : %lf 秒\n", t->binarizing);
	printf("morphology : %lf 秒\n", t->morphology);
	printf("connectedComponent : %lf 秒\n", t->connectedComponent);
	printf("propertyAnalysis : %lf 秒\n", t->propertyAnalysis);
	printf("drawing : %lf 秒\n", t->drawing);

}

int main() {
	//創建資料夾
	char folderName[] = "Image_cv2";

	if (_access(folderName, 0) == -1)
	{
		_mkdir(folderName);
	}


	Time mytime;



	char* path = "hand.bmp";
	Mat src_img = imread(path);

	//灰階圖片黑白轉換
	Mat gray_img = imread(path, 0);;
	gray_img = gray_img ^ 0XFF;
	imwrite("./Image_cv2/1_grayImg.bmp", gray_img);


	// 二值化 Otsu
	clock_t start = clock();
	Mat otsu_img;
	threshold(gray_img, otsu_img, 0, 255, THRESH_OTSU);
	clock_t	end = clock();
	mytime.binarizing = (double)(end - start) / CLK_TCK;
	imwrite("./Image_cv2/2_otsuImg.bmp", otsu_img);

	// 侵蝕、膨脹  morphology
	start = clock();
	dilate(otsu_img, otsu_img, Mat());
	dilate(otsu_img, otsu_img, Mat());
	erode(otsu_img, otsu_img, Mat());
	erode(otsu_img, otsu_img, Mat());
	end = clock();
	mytime.morphology = (double)(end - start) / CLK_TCK;
	imwrite("./Image_cv2/3_morphalogyImg.bmp", otsu_img);

	//連通元件  contour
	start = clock();
	Mat labels, handStats, handsCentroids;
	int handsconcom = connectedComponentsWithStats(otsu_img, labels, handStats, handsCentroids);
	end = clock();
	mytime.connectedComponent = (double)(end - start) / CLK_TCK;
	mytime.propertyAnalysis = (double)(end - start) / CLK_TCK;
	


	// draw box
	start = clock();
	Mat output_img;
	cvtColor(otsu_img, output_img, COLOR_GRAY2BGR);

	cout << "手掌 :\n";
	for (int i = 1; i < handsconcom; i++) {
		int x = handStats.at<int>(i, CC_STAT_LEFT);
		int y = handStats.at<int>(i, CC_STAT_TOP);
		int w = handStats.at<int>(i, CC_STAT_WIDTH);
		int h = handStats.at<int>(i, CC_STAT_HEIGHT);
		
		rectangle(output_img, Rect(x, y, w, h), Scalar(0, 0, 255),2);
		cout << "centroid :"<<"(" << handsCentroids.at<double>(i*2) << "," << handsCentroids.at<double>(i*2+1) << ") , "
			 << "area : " << handStats.at<int>(i, CC_STAT_AREA) << ", ";
		
		if (w > h) {
			cout << "long :" << w << ", orientation: 0度"<<endl;
		}
		else {
			cout << "long :" << h << ", orientation: 90度"<<endl;
		}
	}
	end = clock();
	mytime.drawing = (double)(end - start) / CLK_TCK;

	imshow("handsImg", output_img);
	imwrite("./Image_cv2/4_Hands.bmp", output_img);


	show_time(&mytime);
	//---------------------------------------
	//  Bonus
	//---------------------------------------
	cout << endl;
	Mat bonus_img;
	otsu_img.copyTo(bonus_img);
	Mat mask = getStructuringElement(MORPH_ELLIPSE, Size(24, 24));

	// 侵蝕、膨脹
	erode(bonus_img, bonus_img, mask, Point(-1, -1), 1);
	dilate(bonus_img, bonus_img, mask, Point(-1, -1), 1);
	imwrite("./Image_cv2/5_nofingers.bmp", bonus_img);

	bonus_img = otsu_img - bonus_img;
	erode(bonus_img, bonus_img, Mat(), Point(-1, -1), 1);
	dilate(bonus_img, bonus_img, Mat(), Point(-1, -1), 1);
	imwrite("./Image_cv2/6_fingers.bmp", bonus_img);

	//
	Mat fingerslabels, fingersStats, fingersCentroids;
	int fingersconcom = connectedComponentsWithStats(bonus_img, fingerslabels, fingersStats, fingersCentroids);


	// draw box
	Mat bgr_img;
	cvtColor(bonus_img, bgr_img, COLOR_GRAY2BGR);

	cout << "手指 :\n";
	for (int i = 1; i < fingersconcom; i++) {
		int x = fingersStats.at<int>(i, CC_STAT_LEFT);
		int y = fingersStats.at<int>(i, CC_STAT_TOP);
		int w = fingersStats.at<int>(i, CC_STAT_WIDTH);
		int h = fingersStats.at<int>(i, CC_STAT_HEIGHT);

		rectangle(output_img, Rect(x, y, w, h), Scalar(255, 0, 0), 2);
		cout << "centroid :" << "(" << fingersCentroids.at<double>(i * 2) << "," << fingersCentroids.at<double>(i * 2 + 1) << ") , "
			<< "area : " << fingersStats.at<int>(i, CC_STAT_AREA) << ", ";

		if (w > h) {
			cout << "long :" << w << ", orientation: 0度" << endl;
		}
		else {
			cout << "long :" << h << ", orientation: 90度" << endl;
		}
	}

	//imshow("fingersImg", output_img);
	



	//數手指
	cout << endl;
	int fingers_num;
	for (int i = 1; i < handsconcom; i++) {
		int x = handStats.at<int>(i, CC_STAT_LEFT);
		int y = handStats.at<int>(i, CC_STAT_TOP);
		int w = handStats.at<int>(i, CC_STAT_WIDTH);
		int h = handStats.at<int>(i, CC_STAT_HEIGHT);

		Rect rect(x, y, w, h);
		fingers_num = 0;
		for (int j = 1; j < fingersconcom; j++) {
			int c1 = (int)fingersCentroids.at<double>(j * 2);
			int c2 = (int)fingersCentroids.at<double>(j * 2 + 1);
			if (c1 > x && c1<(x + w) && c2>y && c2 < (y + h)) {
				fingers_num++;
			}
		}
		Point coor;
		coor.x = (int)handsCentroids.at<double>(i * 2);
		coor.y = (int)handsCentroids.at<double>(i * 2 + 1);
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		char text[100];
		sprintf_s(text, "%d", fingers_num);

		putText(output_img, text, coor, font_face, 2, Scalar(0, 0, 255), 3);
		cout << fingers_num<<",";
	}

	imshow("TES2", output_img);
	imwrite("./Image_cv2/7_fingers_boxImg.bmp", output_img);

	waitKey(0);
	destroyAllWindows();
}