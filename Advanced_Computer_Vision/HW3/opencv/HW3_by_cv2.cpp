#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用

using namespace std;

typedef void (*CVT)(cv::Mat&, int*, int*, const int, const int);

cv::Mat birdEyeView(cv::Mat& src);
void findImgSize(cv::Mat& src, int* x1, int* y1, int* x2, int* y2, CVT);
void imgToWorld(cv::Mat& src, int* x, int* y, const int u, const int v);
void worldToImg(cv::Mat& src, int* u, int* v, const int x, const int y);
void hough(cv::Mat& src, cv::Mat& dst, vector<cv::Vec2f>& vec, const int rho, const double theta, const int choose, const int color);
void find_line(cv::Mat& src, cv::Mat& dst);
cv::Mat merge(cv::Mat& up, cv::Mat& down);
cv::Mat repair(cv::Mat& src, cv::Mat& cpy);
//######################################################################################################
int main() {
	//創建資料夾
	char folderName[] = "Image_cv2";

	if (_access(folderName, 0) == -1)
	{
		_mkdir(folderName);
	}


	//圖片路徑
	const char* path = "../WM5fb2261fb1a7b.bmp";
	cv::Mat inputImg = cv::imread(path);

	//crop 
	double alpha = 15 * 3.14159 / 180;
	double calc = ((double)inputImg.rows - 1) * (-0.025 + alpha) / (2 * alpha);
	int uHorizon = (int)calc;
	cv::Mat roi(inputImg, cv::Rect(0, uHorizon, inputImg.cols, inputImg.rows-uHorizon));  //call by reference 不要直接用呀XD
	cv::Mat cropImg;
	roi.copyTo(cropImg);
	cv::imwrite("./Image_cv2/1_crop.bmp", cropImg);
	

	//鳥瞰圖
	cv::Mat birdImg;
	birdImg = birdEyeView(cropImg);
	cv::imwrite("./Image_cv2/2_bird.bmp", birdImg);
	
	//找車道線
	//灰階
	//cv::Mat grayImg = cv::imread(path, 0);
	cv::Mat grayImg = cv::Mat::zeros(cropImg.rows,cropImg.cols,CV_8UC1);
	
	for (int r = 0; r < grayImg.rows; r++) {
		for (int c = 0; c < grayImg.cols; c++) {
			grayImg.at<uchar>(r,c) = 0;
			if (cropImg.at<cv::Vec3b>(r,c)[2]>191) {
				grayImg.at<uchar>(r,c) = 255;
			}
		}
	}

	cv::imwrite("./Image_cv2/3_gray.bmp", grayImg);
	
	cv::Mat sobelImg;
	cv::Sobel(grayImg, sobelImg, 0, 0, 1);
	cv::Mat mask = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));


	// 侵蝕、膨脹
	erode(sobelImg, sobelImg, mask,cv::Point(-1, -1), 1);
	dilate(sobelImg, sobelImg, mask, cv::Point(-1, -1), 1);
	cv::imwrite("./Image_cv2/4_sobel.bmp", sobelImg);
	//hough
	cv::Mat M_houghImg;
	cropImg.copyTo(M_houghImg);
	vector<cv::Vec2f> Mline;
	hough(sobelImg, M_houghImg, Mline, 400,40, 1,2);

	cv::Mat R_houghImg;
	M_houghImg.copyTo(R_houghImg);
	vector<cv::Vec2f> Rline;
	hough(sobelImg, R_houghImg, Mline, 101, 20, 1, 1);


	cv::Mat L_houghImg;
	R_houghImg.copyTo(L_houghImg);
	find_line(sobelImg, L_houghImg);
	cv::imwrite("./Image_cv2/5_line.bmp", L_houghImg);

	//merge
	cv::Mat mergeImg;
	mergeImg = merge(inputImg, L_houghImg);
	cv::imwrite("./Image_cv2/6_all_lines.bmp", mergeImg);

	//鳥瞰2
	cv::Mat bird2Img;
	bird2Img = birdEyeView(L_houghImg);
	cv::imwrite("./Image_cv2/7_Line_bird.bmp", bird2Img);

	//crop
	cv::Mat repairImg;
	repairImg = repair(bird2Img, birdImg);

	cv::Mat roi2(repairImg, cv::Rect(0, 184, 66, 66));  //call by reference 不要直接用呀XD
	cv::Mat crop_img;
	roi2.copyTo(crop_img);
	cv::imwrite("./Image_cv2/8_line_bird_crop.bmp", crop_img);
	

	//imshow("test", crop_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
//######################################################################################################

cv::Mat repair(cv::Mat& src, cv::Mat& cpy) {
	cv::Mat tmp;
	cpy.copyTo(tmp);
	int b, g, red;
	int* lines_rows = (int*)calloc(3, sizeof(int));
	for (int i = 0; i < 3; i++) {
		lines_rows[i] = -1;
	}
	for (int c = 0; c < src.cols; c++) {
		for (int r = 0; r < src.rows; r++) {
			b = src.at<cv::Vec3b>(r, c)[0];
			g = src.at<cv::Vec3b>(r, c)[1];
			red = src.at<cv::Vec3b>(r, c)[2];

			if (b == 255 && g == 0 && red == 0) {
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

	int ro;
	for (int i = 0; i < 3; i++) {
		ro = lines_rows[i];
		for (int c = 0; c < src.cols; c++) {
			tmp.at<cv::Vec3b>(ro, c)[0];
			tmp.at<cv::Vec3b>(ro, c)[1];
			tmp.at<cv::Vec3b>(ro, c)[2];

			if (i == 0) {
				tmp.at<cv::Vec3b>(ro, c)[0] = 255;
			}
			if (i == 1) {
				tmp.at<cv::Vec3b>(ro, c)[1] = 255;
			}
			if (i == 2) {
				tmp.at<cv::Vec3b>(ro, c)[2] = 255;
			}
		}
	}

	

	
	return tmp;
}


cv::Mat merge(cv::Mat& up, cv::Mat& down) {
	cv::Mat output_img = cv::Mat::zeros(up.rows, up.cols, CV_8UC3);



	int r_change, c_change;
	r_change = up.rows - down.rows;

	for (int r = 0; r < up.rows; r++) {
		for (int c = 0; c < up.cols; c++) {
			if (r < r_change) {
				output_img.at<cv::Vec3b>(r, c)[0] = up.at<cv::Vec3b>(r, c)[0];
				output_img.at<cv::Vec3b>(r, c)[1] = up.at<cv::Vec3b>(r, c)[1];
				output_img.at<cv::Vec3b>(r, c)[2] = up.at<cv::Vec3b>(r, c)[2];
			}
			else {
				output_img.at<cv::Vec3b>(r, c)[0] = down.at<cv::Vec3b>(r - r_change, c)[0];
				output_img.at<cv::Vec3b>(r, c)[1] = down.at<cv::Vec3b>(r - r_change, c)[1];
				output_img.at<cv::Vec3b>(r, c)[2] = down.at<cv::Vec3b>(r - r_change, c)[2];
			}
		}
	}

	return output_img;
}


void find_line(cv::Mat& src, cv::Mat& dst) {
	//找左邊線
	float x1, y1, x2, y2;
	x1 = x2 = y1 = y2 = 0;
	int run = 1;
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (src.at<uchar>(c, r) > 0) {
				x1 = r;
				y1 = c;
				run = 0;
				break;
			}
		}
		if (run == 0) {
			break;
		}
	}
	run = 1;
	for (int c = 0; c < src.cols; c++) {
		for (int r = 0; r < src.rows; r++) {
			if (src.at<uchar>(c, r) > 0) {
				x2 = r;
				y2 = c;
				run = 0;
				break;
			}
		}
		if (run == 0) {
			break;
		}
	}
	cv::Point p_start, p_end;

	p_start.x = x1;
	p_start.y = y1;
	p_end.x = x2-(y2*(x1-x2)/(y1-y2));
	p_end.y = y2-y2;
	
	cout << "左  :" << x1 << "," << y1 << "," << x2 << "," << y2 << endl;
	cv::line(dst, p_start, p_end, cv::Scalar(255, 0, 0), 7, cv::LINE_AA);


}



void hough(cv::Mat& src, cv::Mat& dst, vector<cv::Vec2f>& vec, const int rho, const double theta, const int choose,const int color) {
	cv::HoughLines(src, vec, rho, theta* CV_PI / 180, 100);
	const int alpha = 1000;

	if (choose != -1) {
		int i;
		i = choose;
		double x, y;
		double co, si;

		co = (double)cos((float)vec[i][1]);
		si = (double)sin((float)vec[i][1]);
		x = (double)vec[i][0] * co;
		y = (double)vec[i][0] * si;
		
		cv::Point p_start, p_end;

		p_start.x = cvRound(x + alpha * (-si));
		p_start.y = cvRound(y + alpha * (co));
		p_end.x = cvRound(x - alpha * (-si));
		p_end.y = cvRound(y - alpha * (co));

		vec[0][0] = p_start.x;
		vec[0][1] = p_start.y;
		vec[1][0] = p_end.x;
		vec[1][1] = p_end.y;
		
		cv::line(dst, p_start, p_end, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		if (color == 0) {
			cv::line(dst, p_start, p_end, cv::Scalar(255, 0, 0), 7, cv::LINE_AA);
		}
		else if (color == 1) {
			cv::line(dst, p_start, p_end, cv::Scalar(0, 255, 0), 7, cv::LINE_AA);
		}
		else {
			cv::line(dst, p_start, p_end, cv::Scalar(0, 0, 255), 7, cv::LINE_AA);
		}
	}
	else {
		for (size_t i = 0; i < vec.size(); i++) {
			double x, y;
			double co, si;
		
			co = cos(vec[i][1]);
			si = sin(vec[i][1]);
			x = (double)vec[i][0] * co;
			y = (double)vec[i][0] * si;

			cv::Point p_start, p_end;

			p_start.x = cvRound(x + alpha * (-si)) ;
			p_start.y = cvRound(y + alpha * (co)) ;
			p_end.x = cvRound(x - alpha * (-si));
			p_end.y = cvRound(y - alpha * (co));

			
			
			
			if (color == 0) {
				cv::line(dst, p_start, p_end, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
			}
			else if (color == 1) {
				cv::line(dst, p_start, p_end, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			}
			else {
				cv::line(dst, p_start, p_end, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
			}
			
		}
	}
}


void worldToImg(cv::Mat& src, int* u, int* v, const int x, const int y) {
	double alpha = 15 * 3.14159 / 180;
	double dx = 4.0;
	double dy = -10.0;
	double dz = 5.0;
	double y0 = 0.0;
	double th0 = 0.25;
	double m = (double)src.rows;
	double n = (double)src.cols;

	double calculate;
	calculate = (x - dx) / (y - dy);
	calculate = atan(calculate);
	calculate = dz * sin(calculate) / (x - dx);
	calculate = atan(calculate) - th0 + alpha;
	calculate = calculate * (m - 1) / (2 * alpha);
	*u = (int)calculate;

	
	calculate = 0;
	calculate = (x - dx) / (y - dy);
	calculate = atan(calculate) - y0 + alpha;
	calculate = calculate * (n - 1) / (2 * alpha);
	*v = (int)calculate;
}


void imgToWorld(cv::Mat& src, int* x, int* y, const int u, const int v) {
	double alpha = 15 * 3.14159 / 180;
	double dx = 4.0;
	double dy = -10.0;
	double dz = 5.0;
	double y0 = 0.0;
	double th0 = 0.25;
	double m = (double)src.rows;
	double n = (double)src.cols;

	double calculate;
	calculate = 1 / tan(th0 - alpha + (2 * alpha * u) / (m - 1));
	calculate *= dz * sin(y0 - alpha + (2 * alpha * v) / (n - 1));

	*x = (int)(calculate + dx);
	*y = (int)(calculate + dy);
}

void findImgSize(cv::Mat& src, int* x1, int* y1, int*x2, int* y2, CVT cvt) {
	
	int x, y;
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			(*cvt)(src, &x, &y, r, c);
			*x1 = (*x1 > x) ? (x) : (*x1);
			*y1 = (*y1 > y) ? (y) : (*y1);
			*x2 = (*x2 < x) ? (x) : (*x2);
			*y2 = (*y2 < y) ? (y) : (*y2);
		}
	}
}

cv::Mat birdEyeView(cv::Mat& src) {
	// 找world 影像 size
	int x1, y1, x2, y2;
	x1 = y1 = x2 = y2 = 0; //初始化
	findImgSize(src, &x1, &y1, &x2, &y2, imgToWorld);
	cout << x1 << "," << y1 << "," << x2 << "," << y2 << endl;

	//建立 world 影像
	int world_height = (x2 - x1) / 10;
	int world_width = (y2 - y1) / 10;
	cv::Mat bird_img = cv::Mat::zeros(world_height, world_width, CV_8UC3);

	//尋找影像值
	int u, v;
	double scale = 10;
	double X, Y;
	for (int x = 0; x < bird_img.rows; x++) {
		for (int y = 0; y < bird_img.cols; y++) {
			//找 原圖的轉換座標		
			X = (x * 10.0 + x1) / scale;   //scale讓取的範圍在小一點 
			Y = y * 10.0 / scale;
			worldToImg(src, &u, &v, X, Y);

			if (u >= 0 && v >= 0) {
				if (u < src.rows && v < src.cols) {
					for (int i = 0; i < 3; i++) {
						bird_img.at<cv::Vec3b>(x, y)[i] = src.at<cv::Vec3b>(u, v)[i];
					}
				}
			}
		}
	}

	return bird_img;
}