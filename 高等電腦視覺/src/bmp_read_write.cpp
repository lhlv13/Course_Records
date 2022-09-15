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



Mat imread(const char* path);
void imwrite(const char* path, Mat img);

int main(void) {
	const char* path = "./lena.bmp";
	Mat img;

	img = imread(path);
	imwrite("./test.bmp", img);
}


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