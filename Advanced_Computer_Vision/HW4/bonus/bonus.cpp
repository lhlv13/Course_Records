#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include<math.h>
#include <io.h>   //創 img資料夾用
#include <direct.h>  //創 img資料夾用
#include<ctime>

using namespace std;
using namespace cv;

Mat template_matching(Mat& pre, Mat& now, Mat& colorImg, Mat* output_grayImg, Mat* output_difImg);
double snr(const Mat& I1, const Mat& I2);
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


	const char* src = "../WM61b1c81394b01.avi";   // 修改輸入影片路徑
	Mat colorImg, grayImg, preImg, nowImg;
	Mat outputImg;
	Mat output_grayImg;
	Mat output_difImg;
	int count = 0;
	double SNR;
	double avg_SNR=0, total_opf_time=0, total_temp_time=0;
	char text[80];

	VideoCapture capture(src);
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "影片路徑錯誤!!" << endl;
		return 0;
	}
	Mat frame1, prvs;
	capture >> frame1;
	cvtColor(frame1, prvs, COLOR_BGR2GRAY);

	//影片儲存初始化
	Size s = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
	int fps = capture.get(CAP_PROP_FPS);
	VideoWriter writer("./output_mp4/color.mp4", CAP_OPENCV_MJPEG, fps, s, true);
	VideoWriter writer_gray("./output_mp4/gray.mp4", CAP_OPENCV_MJPEG, fps, s, true);

	while (true) {
		clock_t opf_t_start = clock();
		Mat frame2, next;
		capture >> frame2;
		if (frame2.empty())
			break;
		cvtColor(frame2, next, COLOR_BGR2GRAY);
		Mat flow(prvs.size(), CV_32FC2);
		calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		// visualization
		Mat flow_parts[2];
		split(flow, flow_parts);
		Mat magnitude, angle, magn_norm;
		cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
		normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
		angle *= ((1.f / 360.f) * (180.f / 255.f));
		//build hsv image
		Mat _hsv[3], hsv, hsv8, bgr;
		_hsv[0] = angle;
		_hsv[1] = Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magn_norm;
		merge(_hsv, 3, hsv);
		hsv.convertTo(hsv8, CV_8U, 255.0);
		cvtColor(hsv8, bgr, COLOR_HSV2BGR);
		clock_t opf_t_end = clock();


		//影像處理
		
		cvtColor(bgr, grayImg, COLOR_BGR2GRAY);
		
		grayImg.copyTo(nowImg);
		if (count > 0) {

			// 影像處理
			clock_t temp_t_start = clock();
			outputImg = template_matching(preImg, nowImg, colorImg, &output_grayImg, &output_difImg);
			clock_t temp_t_end = clock();

			//計算SNR
			SNR = snr(nowImg, output_grayImg);

			avg_SNR += SNR;
			total_opf_time += (double)(opf_t_end - opf_t_start) / CLK_TCK;
			total_temp_time += (double)(temp_t_end - temp_t_start) / CLK_TCK;
			cout << "Frame = " << count + 1 << " , " << "SNR = " << SNR << " , ";
			cout << "光流法花費時間 : " << (double)(opf_t_end - opf_t_start) / CLK_TCK << "(s) , 模板匹配花費時間 : " 
				<< (double)(temp_t_end - temp_t_start) / CLK_TCK <<"(s)"<< endl;

			//輸出影片
			imshow("video", outputImg);
			waitKey(10);


			//儲存影片
			writer.write(outputImg);
			writer_gray.write(output_grayImg);

			//儲存照片
			sprintf_s(text, "./gray_images/%d_%d.bmp", count, count + 1);
			imwrite(text, output_grayImg);
			sprintf_s(text, "./differ_images/%d.bmp", count + 1);
			imwrite(text, output_difImg);
			sprintf_s(text, "./color_images/%d_%d.bmp", count, count + 1);
			imwrite(text, outputImg);

		}

		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
		prvs = next;
		count++;
		nowImg.copyTo(preImg);
		bgr.copyTo(colorImg);
	}
	cout <<endl<<endl<< "平均 SNR = " << avg_SNR/(count-1) << " , ";
	cout << "光流法花費總時間 : " << total_opf_time << "(s) , 模板匹配花費總時間 : "
		<< total_temp_time << "(s)" << endl;
	capture.release();
	writer.release();
	writer_gray.release();

	system("pause");
	return 0;
}



double snr(const Mat& I1, const Mat& I2) {
	int n = I1.rows * I1.cols;
	Scalar avg_ = sum(I1) / n;

	Mat diff_img;
	absdiff(I2, I1, diff_img);
	Scalar avg_n = sum(diff_img) / n;

	double vs = 0;
	double vn = 0;

	for (int r = 0; r < I1.rows; r++) {
		for (int c = 0; c < I1.cols; c++) {
			vs += pow((I1.at<uchar>(r, c) - avg_[0]), 2);
			vn += pow((I2.at<uchar>(r, c) - I1.at<uchar>(r, c) - avg_n[0]), 2);
		}
	}
	vs = vs / n;
	vn = vn / n;
	double snr = 20 * log10((pow(vs, 0.5), pow(vn, 0.5)));

	return snr;
	
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
			for (int r = 0; r < 40; r++) {
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