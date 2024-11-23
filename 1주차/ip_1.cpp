//2020037049 김세호
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>

using namespace cv; //cv::쓸 때 cv:: 안써도 됨

typedef struct {
	int r, g, b;
}int_rgb;

int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}

float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imwrite(name, img);
}

void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0        Ե        

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

int ex0903() {
	Mat img = imread("anq.png");

	imshow("test", img);
	waitKey(0);

	printf("Hello World\n");
	return 0;
}

void ex0911_1() { //십자모양 만들기
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width); //x 방향으로 256, y 방향으로 512 크기의 2차원 배열 생성
	//모든 컬러 값은 0으로 초기화

	int y = 256;
	int x = 512;

	for (x = 0; x < width; x++) {
		img[y][x] = 255;
	}

	x = 512;
	for (y = 0; y < height; y++) {
		img[y][x] = 255;
	}

	ImageShow((char*)"output", img, height, width); //내부에 imshow 함수가 있음
	
	IntFree2(img, 256, 512); //메모리 해제
}

void drawLine(int** imgxx, int y, int x0,int x1) { //외부 함수 선언
	//int** imgxx, int y, int x0,int x1 //와 같은 문장임
	for (int x = x0; x < x1; x++) {
		imgxx[y][x] = 255; //y, img는 밖에서 넘겨줘야 됨
	}
}

void ex0911_2() { //중간에 네모 만들기
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width); //x 방향으로 256, y 방향으로 512 크기의 2차원 배열 생성
	//모든 컬러 값은 0으로 초기화

	int y = 256;
	int x = 512;
	
	drawLine(img, y, width/3, height/7); //함수 호출
	//사각형 그리려면 for문에 넣어서 돌리면 나옴
	for (y = 20; y < 40; y++) {
		drawLine(img, y, 300, 812);
	}

	for (x = 0; x < width; x++) {
		for (y = 0; y < height; y++) {
			if ((x > 256 && x < 768) && (y > 128 && y < 384)) { //조건문 다 성립할 때만 해당 픽셀값 255로 바꿈
				img[y][x] = 255; 
			}
		}
	}

	ImageShow((char*)"output", img, height, width); //내부에 imshow 함수가 있음

	IntFree2(img, 256, 512); //메모리 해제
}

void ex0911_3() { //중간에 원 만들기
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width); //x 방향으로 256, y 방향으로 512 크기의 2차원 배열 생성
	//모든 컬러 값은 0으로 초기화

	int y = 256;
	int x = 512;

	for (x = 0; x < width; x++)
		for (y = 0; y < height; y++) {
			if ((x - 512) * (x - 512) + (y - 256) * (y - 256) < 10000) //원의 방정식 x^2 + y^2 = r^2
			//10000은 원의 반지름의 제곱(여기선 100^2)
			img[y][x] = 255; //조건문 다 성립할 때만 해당 픽셀값 255로 바꿈
		}

	WriteImage((char*)"test.jpg", img, height, width);
	ImageShow((char*)"output", img, height, width); //내부에 imshow 함수가 있음
	IntFree2(img, 256, 512); //메모리 해제
}

void Thresholding(int threshold, int** img, int height, int width, int** img_out){
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] > threshold)
				img_out[y][x] = 255;
			else
				img_out[y][x] = 0;
		}
	}
}

void ex0924_1() {
   int height, width;
   int** img = ReadImage((char*)"./TestImages/lenax0.5.png", &height, &width);
   int** img_out = (int**)IntAlloc2(height, width);
   int threshold = 128;

   ImageShow((char*)"input", img, height, width); //기본 이미지

   for (threshold = 50; threshold < 250; threshold += 50) {
	   Thresholding(threshold, img, height, width, img_out); //임계값 처리
	   ImageShow((char*)"output", img_out, height, width); //임계값 처리된 이미지
	   //for문 돌려서 여러 임계값 처리된 이미지를 보여줌
   }
}

void ShiftImage(int value, int** img, int height, int width)
 {
	 for (int y = 0; y < height; y++) {
		 for (int x = 0; x < width; x++) {
			 img[y][x] = img[y][x] + value;
		 }
	 }
	 //1의 보수는 0은 1로, 1은 0으로 바꾸는 것이다.
	 //2의 보수는 1의 보수에 1을 더한 것이다.
	 //컴퓨터는 2의 보수를 사용하는데, 1 = 0000 0001, -1 = 1111 11111로 표현되기 때문에
	 //8비트 시스템에선 음수는 매우 밝게 표현된다.
 }

#define GetMax(x,y) ((x>y)?x:y)
#define GetMin(x,y) ((x<y)?x:y)

void ClippingImage(int** img_out, int** img, int height, int width) {
	 for (int y = 0; y < height; y++) { 
		 for (int x = 0; x < width; x++) {
			 //if (img[y][x] > 255) {
				// img_out[y][x] = 255; //255보다 크면 255로
			 //}
			 //else if (img[y][x] < 0) {
				// img_out[y][x] = 0; //0보다 작으면 0으로
			 //}
			 //else {
				// img_out[y][x] = img[y][x]; //그 외에는 그대로
			 //}

			 //위의 코드를 매크로를 사용해 바꿈
			 /*int A = GetMax(img[y][x], 0);
			 int B = GetMin(A, 255);
			 img_out[y][x] = B;*/
			 
			 //위의 코드를 한 줄로 바꿈
			 img_out[y][x] = GetMin(GetMax(img[y][x], 0), 255);

			 //위의 3 문단은 다 같은 기능을 수행하는 코드
		 }
	 }
	 //클리핑은 특정 범위를 벗어나면 그 값을 최대값 또는 최소값으로 바꾸는 것이다.
	 //예를 들어 0~255 사이의 값만 허용하고 싶을 때 사용한다.
 }

void ex0924_2()
 {
	 int height, width;
	 int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	 int** img_out = (int**)IntAlloc2(height, width);

	 ShiftImage(50, img, height, width); 
	 ClippingImage(img_out, img, height, width);

	 
	 ImageShow((char*)"input", img, height, width); 
	 ImageShow((char*)"output", img_out, height, width);
 }

 //매크로
#define NUM 300

void ex0924_3() {
	 int aaa = 100, bbb = NUM;
	 int A, B;
	 //a = (x > y) ? x : y; //a에 x가 y보다 크면(참이면) x를 넣고 아니면(거짓이면) y를 넣어라
	 //b = (x < y) ? x : y; //b에 x가 y보다 작으면(참이면) x를 넣고 아니면(거짓이면) y를 넣어라

	 A = GetMax(aaa, 0); //A = ((aaa > 0) ? aaa : 0);
	 B = GetMin(bbb, 255); //B = ((bbb < 255) ? bbb : 255);
 }

//중간값 구하는 매크로도 만들 수 있음
#define GetMid(x, y, z) ((x > y) ? ((y > z) ? y : ((x > z) ? z : x)) : ((x > z) ? x : ((y > z) ? z : y)))

void ex0925_1() {
	 int A = 100, B = 200, C = 300;
	 /*int D = GetMax(A, B);
	 int E = GetMax(D, C);*/

	 int E = GetMax(GetMax(A, B), C); //최댓값 구하는 매크로를 사용해 한 줄로 표현
 }

int FindMaxValue(int** img, int height, int width) {
	 int max_value = img[0][0];

	 for (int y = 0; y < height; y++) {
		 for (int x = 0; x < width; x++) {
			 max_value = GetMax(max_value, img[y][x]); //245
		 }
	 }
	 return max_value;
 }

int FindMinValue(int** img, int height, int width) {
	 int min_value = img[0][0];

	 for (int y = 0; y < height; y++) {
		 for (int x = 0; x < width; x++) {
			 min_value = GetMin(min_value, img[y][x]); //245
		 }
	 }
	 return min_value;
 }

void ex0925_2() {
	 int A[7] = { 1,-1,3,8,2,9,10 };
	 int max_value = A[0];

	 /*max_value = GetMax(max_value, A[1]);
	 max_value = GetMax(max_value, A[2]);
	 max_value = GetMax(max_value, A[3]);
	 max_value = GetMax(max_value, A[4]);
	 max_value = GetMax(max_value, A[5]);
	 max_value = GetMax(max_value, A[6]);*/
	 //최댓값 구하는 과정 가시화

	 for (int i = 1; i < 7; i++) {
		 max_value = GetMax(max_value, A[i]);
	 }
	 int height, width;
	 int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);

	 max_value = FindMaxValue(img, height, width); //245
	 int min_value = FindMinValue(img, height, width); //25
}

 //이미지 병합, 영상혼합
void MixingImages(
	int alpha, //in 가중치
	int** img1, //in 첫 번째 이미지
	int** img2, //in 두 번째 이미지
	int height, //in 이미지 높이
	int width, //in 이미지 너비
	int** img_out //out 출력 이미지
) {
	 for (int y = 0; y < height; y++) {
		 for (int x = 0; x < width; x++) {
			 img_out[y][x] = alpha * img1[y][x] + (1 - alpha) * img2[y][x];
		 }
	 }
}

void Ex0925_3()
{
	int height, width;
	int** img1 = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img2 = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float alpha = 0.7;

	for (alpha = 0.1; alpha <= 1.0; alpha += 0.1) {
		MixingImages(alpha, img1, img2, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}
}

void sample() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void Stretch1(int** img, int** img_out, int a, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a) {
				img_out[y][x] = (255.0 / a) * img[y][x] + 0.5;
				//255.0은 실수형으로 나누기 위함
				//255/a 앞에 (float) 넣어도 int 형으로 그대로 출력됨
				//같은 괄호 안에 넣으면 캐스팅 가능
				//255를 255.0으로 바꾸면 실수형으로 계산됨
				//안 하면 정수는 소수점 아래 숫자를 다 버린다.
				//+0.5는 반올림을 위함
			}
			else {
				img_out[y][x] = 255;
			}
		}
	}
}

void Stretch2(
	int a, int b, int c, int d,
	int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] <= a) {
				img_out[y][x] = ((float)c / a) * img[y][x] + 0.5;
				//(float)(255.0 / a) 이런식으로 하면 정수나온다 
				// 나온답에 실수를 붙이는 거기 때문에. 안하는 게 좋다.   
			}
			else if (img[y][x] > a && img[y][x] < b) {
				img_out[y][x] = ((float)d - c) / (b - a) * (img[y][x] - a) + c + 0.5;
			}
			else if (img[y][x] >= b) {
				img_out[y][x] = ((float)(255 - d) / (255 - b)) * (img[y][x] - b) + d + 0.5;
			}
			else {
				img_out[y][x] = 255;
			}
		}

	}
}

struct Parameter {
	int a;
	int b;
	int c;
	int d;
};

void Stretch3(
	Parameter param,
	int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] <= param.a) {
				img_out[y][x] = ((float)param.c / param.a) * img[y][x] + 0.5;
				//(float)(255.0 / a) 이런 식으로 하면 정수나온다 나온답에 실수를 붙이는 거기 때문에. 안하는 게 좋다.   
			}
			else if (img[y][x] > param.a && img[y][x] < param.b) {
				img_out[y][x] = ((float)param.d - param.c) / (param.b - param.a) * (img[y][x] - param.a) + param.c + 0.5;
			}
			else if (img[y][x] >= param.b) {
				img_out[y][x] = ((float)(255 - param.d) / (255 - param.b)) * (img[y][x] - param.b) + param.d + 0.5;
			}
			else {
				img_out[y][x] = 255;
			}
		}
	}
}
struct ParameterAll {
	int a, b, c, d;
	int** img;
	int height, width;
	int** img_out;
};

void Stretch4(ParameterAll param) {
	for (int y = 0; y < param.height; y++) {
		for (int x = 0; x < param.width; x++) {
			if (param.img[y][x] <= param.a) {
				param.img_out[y][x] = ((float)param.c / param.a) * param.img[y][x] + 0.5;
				//(float)(255.0 / a) 이런식으로 하면 정수나온다 나온답에 실수를 붙이는 거기 때문에. 안하는 게 좋다.   
			}
			else if (param.img[y][x] > param.a && param.img[y][x] < param.b) {
				param.img_out[y][x] = ((float)param.d - param.c) / (param.b - param.a) * (param.img[y][x] - param.a) + param.c + 0.5;
			}
			else if (param.img[y][x] >= param.b) {
				param.img_out[y][x] = ((float)(255 - param.d) / (255 - param.b)) * (param.img[y][x] - param.b) + param.d + 0.5;
			}
			else {
				param.img_out[y][x] = 255;
			}
		}
	}
}

void Ex1002_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	//int a = 50;

	//Stretch1(img, img_out, a, height, width);
	
	//Stretch2(a, b, c, d, img, img_out, height, width);

	/*struct Parameter param = { 50, 100, 150, 200 };
	param.a = 100;
	param.b = 150;
	param.c = 50;
	param.d = 200;
	Stretch3(param, img, img_out, height, width);*/

	ParameterAll param;
	param.a = 100; param.b = 150; param.c = 50; param.d = 200;
	param.img = img;
	param.height = height;
	param.width = width;
	param.img_out = img_out;
	//Stretch4({ 100, 150, 50, 200, img, height, width, img_out });
	Stretch4(param);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

//반올림 설명용
void ex_roundup() {
	float a = 100.5; 
	int b = a; //b=100
	b = a + 0.5; //b=101
}

//1008 수업
int GetCount(int height, int width, int value,int** img) { 
	int count = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] == value) {
				count++;
			}
		}
	}
	return count; //픽셀값이 value인 픽셀의 개수 반환
}

void GetHistogram(int height, int width, int* histogram, int** img) { //히스토그램은 주소로 받아야 함(배열의 주소)
	for (int i = 0; i < 256; i++) { //i = value
		histogram[i] = GetCount(height, width, i, img);
	}
}

void ex1008_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);

	int value = 100;
	int count = 0;

	int histogram[256]; //히스토그램 배열 생성

	GetHistogram(height, width, histogram, img); //히스토그램 생성

	DrawHistogram((char*)"histogram", histogram); //히스토그램 출력

	printf("\n count = %d", histogram[value]); //픽셀값이 100인 픽셀의 개수 출력
}

void GetHistogram2(int height, int width, int* histogram, int** img) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			histogram[img[y][x]]++; //histogram을 count 변수 처럼 사용
			//ex1008_1()과 같은 결과지만 최소 256배 빠름
		}
	}
}

void GetChistogram(int height, int width, int** img, int* chist) {
	int histogram[256] = { 0, }; //히스토그램 배열 생성

	GetHistogram2(height, width, histogram, img); //히스토그램 생성

	chist[0] = histogram[0]; //누적 히스토그램
	for (int n = 1; n < 256; n++) {
		chist[n] = chist[n - 1] + histogram[n]; //n=1,2,...,255
	}
	DrawHistogram((char*)"histogram", histogram); //히스토그램 출력
}

void ex1008_2() { //누적 히스토그램
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);

	int value = 100;
	int count = 0;

	int chist[256] = { 0, }; //누적 히스토그램 배열 생성

	GetChistogram(height, width, img, chist); //누적 히스토그램 생성

	DrawHistogram((char*)"chistogram", chist); //히스토그램 출력
}

//히스토그램 평활화 함수화
void Hist_Equalization(int height, int width, int** img, int** img_out, int* chist) {
	int norm_chist[256] = { 0, };
	for (int n = 0; n < 256; n++) {
		norm_chist[n] = (float)chist[n] / (width * height) * 255 + 0.5;
		//누적 히스토그램을 전체 픽셀 수로 나누면 정규화된 누적 히스토그램이 된다.
	}
	//DrawHistogram((char*)"chistogram", chist); //히스토그램 출력

	//mapping using 'norm_chist[]'
	//(img_out[y][x]=)y = f(x)(=norm_chist[img[y][x]])
	norm_chist[0] = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = norm_chist[img[y][x]];
			//정규화된 누적 히스토그램을 사용해 이미지 매핑
		}
	}
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	int hist_input[256] = { 0, };
	int hist_output[256] = { 0, };

	GetHistogram2(height, width, hist_input, img); //input 히스토그램 생성
	GetHistogram2(height, width, hist_output, img_out); //output 히스토그램 생성

	DrawHistogram((char*)"input_hist", hist_input);
	DrawHistogram((char*)"output_hist", hist_output);

	//return은 img_out에 다 들어 있어서 굳이 return 필요 X
}

void Ex1008_3() { //히스토그램 평활화
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lenax0.5.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int value = 100;
	int count = 0;

	int chist[256] = { 0, }; //누적 히스토그램 배열 생성

	GetChistogram(height, width, img, chist); //누적 히스토그램 생성

	Hist_Equalization(height, width, img, img_out, chist); //히스토그램 평활화
}

int getMean3x3(int y, int x, int** img) {
	int sum = 0;
	//평균필터 사이즈가 3*3일 때
	for (int m = -1; m <= 1; m++) {
		for (int n = -1; n <= 1; n++) {
			sum += img[y + m][x + n]; //주변 3x3 픽셀의 합
		}
	}
	return (int)(sum / 9.0 + 0.5);
}

void MeanFilter3x3(int** img, int height, int width, int** img_out) {
	int x, y;
	
	for (y = 1; y < height - 1; y++) { //y는 1열 수평선
		for (x = 1; x < width - 1; x++){  //x는 1행 수직선
			//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
			img_out[y][x] = getMean3x3(y, x, img);
		}
	}

	for (x = 0; x < width; x++) img_out[y][x] = img[y][x];

	y = height - 1;
	for (x = 0; x < width; x++) img_out[y][x] = img[y][x];

	x = 0;
	for (y = 0; y < height; y++) img_out[y][x] = img[y][x];

	x = width - 1;
	for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	//맨 위, 아래, 왼쪽, 오른쪽 테두리만 바로 안쪽 테두리에서 복사

	/*if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
		img_out[y][x] = img[y][x];
	}*/

	/*img_out[y][x] = (img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
		+ img[y][x - 1] + img[y][x] + img[y][x + 1]
		+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0 + 0.5;*/

	//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
	//메모리 액세스 위반이 발생한다.
}

int getMean5x5(int y, int x, int** img) {
	int sum = 0;
	for (int m = -2; m <= 2; m++) {
		for (int n = -2; n <= 2; n++) {
			sum += img[y + m][x + n]; //주변 5X5 픽셀의 합
		}
	}
	 
	return (int) (sum / 25.0 + 0.5);
}

void MeanFilter5x5(int** img, int height, int width, int** img_out) {
	int x, y;

	for (y = 2; y < height - 2; y++) { //y는 1열 수평선
		for (x = 2; x < width - 2; x++) {  //x는 1행 수직선
			//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
			//맨 위, 아래, 왼쪽, 오른쪽 테두리만 바로 안쪽 테두리에서 복사
			img_out[y][x] = getMean5x5(y, x, img);
		}
		/*if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				img_out[y][x] = img[y][x];
			}*/
	}
	for (y = 0; y < 2; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (y = height - 2; y < height; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (x = 0; x < 2; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}

	for (x = width - 2; x < width; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}
	//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
	//메모리 액세스 위반이 발생한다.
}

int getMean7x7(int y, int x, int** img) {
	int sum = 0;
	for (int m = -3; m <= 3; m++) {
		for (int n = -3; n <= 3; n++) {
			sum += img[y + m][x + n]; //주변 5X5 픽셀의 합
		}
	}
	return (int)(sum / 49.0 + 0.5);
}

void MeanFilter7x7(int** img, int height, int width, int** img_out) {
	int x, y;

	for (y = 3; y < height - 3; y++) { //y는 1열 수평선
		for (x = 3; x < width - 3; x++) { //x는 1행 수직선
			//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
			//맨 위, 아래, 왼쪽, 오른쪽 테두리만 바로 안쪽 테두리에서 복사
			img_out[y][x] = getMean7x7(y, x, img);
		}
		/*if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				img_out[y][x] = img[y][x];
			}*/
	}
	for (y = 0; y < 3; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (y = height - 3; y < height; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (x = 0; x < 3; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}

	for (x = width - 3; x < width; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}
	//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
	//메모리 액세스 위반이 발생한다.
}
	
void ex1015_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out3x3 = (int**)IntAlloc2(height, width);
	int** img_out5x5 = (int**)IntAlloc2(height, width);
	int** img_out7x7 = (int**)IntAlloc2(height, width);

	MeanFilter3x3(img, height, width, img_out3x3);
	MeanFilter5x5(img, height, width, img_out5x5);
	MeanFilter7x7(img, height, width, img_out7x7);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output3x3", img_out3x3, height, width);
	ImageShow((char*)"output5x5", img_out5x5, height, width);
	ImageShow((char*)"output7x7", img_out7x7, height, width);
}

int getMeanNxN(int N, int y, int x, int** img) {
	int delta = (N - 1) / 2;

	int sum = 0;
	for (int m = -delta; m <= delta; m++) {
		for (int n = -delta; n <= delta; n++) {
			sum += img[y + m][x + n];
		}
	}
	return (int)((float)sum / (N*N) + 0.5); 
}

void MeanFilterNxN(int N, int** img, int height, int width, int** img_out) {
	int delta = (N - 1) / 2; //일반화
	int x, y;

	for (y = delta; y < height - delta; y++) { //y는 1열 수평선
		for (x = delta; x < width - delta; x++) {  //x는 1행 수직선
			//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
			//맨 위, 아래, 왼쪽, 오른쪽 테두리만 바로 안쪽 테두리에서 복사
			img_out[y][x] = getMeanNxN(N, y, x, img);
		}
		/*if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				img_out[y][x] = img[y][x];
			}*/
	}
	for (y = 0; y < delta; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (y = height - delta; y < height; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (x = 0; x < delta; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}

	for (x = width - delta; x < width; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}
	//y=0, x=0일 때는 픽셀의 위치가 범위 밖이라서 에러가 난다.
	//메모리 액세스 위반이 발생한다.
}

//중간고사
// 
//---------------------------------------------------------
// 
//기말고사

void ex1016_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	ImageShow((char*)"input", img, height, width);

	for (int N = 3; N < 15; N += 2) {
		MeanFilterNxN(N, img, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}
}

int MaskingOne(int y, int x, float** kernel, int** img) {
	float sum = 0;

	for (int m = -1; m <= 1; m++) {
		for (int n = -1; n <= 1; n++) {
			sum += img[y + m][x + n] * kernel[m + 1][n + 1]; //마스킹 연산
		}
	}
	return (int)(sum + 0.5);
}

void MaskingImage(int height, int width, int** img, float** kernel, int** img_out) {
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			img_out[y][x] = MaskingOne(y, x, kernel, img); //마스킹 연산
		}
	}
}

void ex1030() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

	kernel[0][0] = 1 / 9.0; kernel[0][1] = 1 / 9.0; kernel[0][2] = 1 / 9.0;
	kernel[1][0] = 1 / 9.0; kernel[1][1] = 1 / 9.0; kernel[1][2] = 1 / 9.0;
	kernel[2][0] = 1 / 9.0; kernel[2][1] = 1 / 9.0; kernel[2][2] = 1 / 9.0;

	MaskingImage(height, width, img, kernel, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void MagGradient_X(int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width-1; x++) {
			img_out[y][x] = abs(img[y][x + 1] - img[y][x]); //x방향 기울기
		}
	}
}

void MagGradient_Y(int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height-1; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = abs(img[y + 1][x] - img[y][x]); //y방향 기울기
		}
	}
}

void ScalingImage(float scale, int** img_in_out,int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width - 1; x++) {
			img_in_out[y][x] = scale * img_in_out[y][x];
			if (img_in_out[y][x] > 255) {
				img_in_out[y][x] = 255;
			}
			else if (img_in_out[y][x] < 0) {
				img_in_out[y][x] = 0;
			}
		}
	}
}

void ex1029_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
	int** img_out_x = (int**)IntAlloc2(height, width);
	int** img_out_y = (int**)IntAlloc2(height, width);

	int y = 200, x = 200;

	MagGradient_X(img, height, width, img_out_x); //수평방향 그레디언트
	MagGradient_Y(img, height, width, img_out_y); //수직방향 그레디언트
	 
	ScalingImage(4, img_out_x, height, width);//그레디언트 값이 너무 작아서 화면에 보이지 않음
	ScalingImage(4, img_out_y, height, width);//따라서 그레디언트 값 4배로 증폭
	
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out_x, height, width);
	ImageShow((char*)"output", img_out_y, height, width);
}

void MagGradient_XY(int** img, int height, int width, int** img_out) { 
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int fy = img[y][x + 1] - img[y][x]; //수직방향 그레디언트
			int fx = img[y + 1][x] - img[y][x]; //수평방향 그레디언트
			img_out[y][x] = abs(fx) + abs(fy); //그레디언트 크기
		}
	}
}

void ex1029_2() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	MagGradient_XY(img, height, width, img_out);
	ScalingImage(2, img_out, height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void ex1030_2() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

	kernel[0][0] = -1; kernel[0][1] = -1; kernel[0][2] = -1; //수직방향
	kernel[1][0] = -1; kernel[1][1] = -8.0; kernel[1][2] = -1; 
	kernel[2][0] = -1; kernel[2][1] = -1; kernel[2][2] = -1;

	MaskingImage(height, width, img, kernel, img_out); //수직방향 마스킹
	ClippingImage(img_out, img, height, width); //클리핑

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void NormalizeImage(int** input, int height, int width, int** output) { 
	int max_value = FindMaxValue(input, height, width); //최대값 찾기

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			output[y][x] = (float)input[y][x] / max_value * 255; //정규화
		}
}

void ex1030_3() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	MagGradient_XY(img, height, width, img_out); 
	
	NormalizeImage(img_out, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

//평균필터(LPF) : 이미지의 노이즈를 제거하는 필터

//백색잡음(White Noise) : 모든 주파수를 가진 잡음
//잡음 : 모든 주파수를 가지고 있는 신호
//백색잡음은 모든 주파수에서 같은 에너지를 가진다

void AbsImage(int** img_in, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = abs(img_in[y][x]); //절대값
		}
	}
}

void ex1105_1() { //라플라시안	필터
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

#if 0
	kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0;
	kernel[1][0] = -1; kernel[1][1] = 4.0; kernel[1][2] = -1;
	kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0;

#else
	kernel[0][0] = -1.0; kernel[0][1] = -1.0; kernel[0][2] = -1.0;
	kernel[1][0] = -1.0; kernel[1][1] = 8.0; kernel[1][2] = -1.0;
	kernel[2][0] = -1.0; kernel[2][1] = -1.0; kernel[2][2] = -1.0;
#endif

	MaskingImage(height, width, img, kernel, img_out); //마스킹
	AbsImage(img_out, height, width, img_out); //절대값(앞에 img_out이 input)
	//ClippingImage(img_out, img_out, height, width); //클리핑
	NormalizeImage(img_out, height, width, img_out); //정규화

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void ex1105_2() { //소벨 필터
	int height, width;
	int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

#if 0 //가로방향
	kernel[0][0] = -1.0; kernel[0][1] = -2; kernel[0][2] = -1;
	kernel[1][0] = 0; kernel[1][1] = 0; kernel[1][2] = 0;
	kernel[2][0] = 1.0; kernel[2][1] = 2; kernel[2][2] = 1;

#else //세로방향
	kernel[0][0] = 1.0; kernel[0][1] = 0; kernel[0][2] = -1.0;
	kernel[1][0] = 2; kernel[1][1] = 0; kernel[1][2] = -2.0;
	kernel[2][0] = 1.0; kernel[2][1] = 0; kernel[2][2] = -1.0;
#endif

	MaskingImage(height, width, img, kernel, img_out); //마스킹
	AbsImage(img_out, height, width, img_out); //절대값(앞에 img_out이 input)
	//ClippingImage(img_out, img_out, height, width); //클리핑
	NormalizeImage(img_out, height, width, img_out); //정규화

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void MagSobel_X(int** img, int height, int width, int** img_out) {
	float** kernel = (float**)FloatAlloc2(3, 3);

	kernel[0][0] = -1.0; kernel[0][1] = -2; kernel[0][2] = -1;
	kernel[1][0] = 0; kernel[1][1] = 0; kernel[1][2] = 0;
	kernel[2][0] = 1.0; kernel[2][1] = 2; kernel[2][2] = 1;

	MaskingImage(height, width, img, kernel, img_out); //마스킹
	AbsImage(img_out, height, width, img_out); //절대값(앞에 img_out이 input)

	FloatFree2(kernel, 3, 3);
}

void MagSobel_Y(int** img, int height, int width, int** img_out) {
	float** kernel = (float**)FloatAlloc2(3, 3);

	kernel[0][0] = -1.0; kernel[0][1] = 0; kernel[0][2] = 1.0;
	kernel[1][0] = -2; kernel[1][1] = 0; kernel[1][2] = 2.0;
	kernel[2][0] = -1.0; kernel[2][1] = 0; kernel[2][2] = 1.0;

	MaskingImage(height, width, img, kernel, img_out); //마스킹
	AbsImage(img_out, height, width, img_out); //절대값(앞에 img_out이 input)

	FloatFree2(kernel, 3, 3); //메모리 해제 안 해주면 프로그램 죽음
}

void MagSobel_XY(int** img, int height, int width, int** img_out_xy) {
	int** img_out_x = (int**)IntAlloc2(height, width);
	int** img_out_y = (int**)IntAlloc2(height, width);

	MagSobel_X(img, height, width, img_out_x); //수평방향 그레디언트
	MagSobel_Y(img, height, width, img_out_y); //수직방향 그레디언트

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			img_out_xy[y][x] = img_out_x[y][x] + img_out_y[y][x]; //x방향, y방향 그레디언트의 합
		}
	IntFree2(img_out_x, height, width);
	IntFree2(img_out_y, height, width);
}

void ex1105_3() { //소벨 필터
	int height, width;
	int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
	int** img_out_xy = (int**)IntAlloc2(height, width);

	MagSobel_XY(img, height, width, img_out_xy);

	NormalizeImage(img_out_xy, height, width, img_out_xy); //정규화

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out_xy, height, width);
}

void ex1106_1() { //선명화 처리
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

#if 1
	kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0;
	kernel[1][0] = -1; kernel[1][1] = 5.0; kernel[1][2] = -1;
	kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0;

#else
	kernel[0][0] = -1.0; kernel[0][1] = -1.0; kernel[0][2] = -1.0;
	kernel[1][0] = -1.0; kernel[1][1] = 9.0; kernel[1][2] = -1.0;
	kernel[2][0] = -1.0; kernel[2][1] = -1.0; kernel[2][2] = -1.0;
#endif

	MaskingImage(height, width, img, kernel, img_out); //마스킹
	ClippingImage(img_out, img_out, height, width); //클리핑

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void GetKernel_S1(float alpha, float** kernel) {
	kernel[0][0] = 0; kernel[0][1] = -alpha; kernel[0][2] = 0;
	kernel[1][0] = -alpha; kernel[1][1] = 1 + 4.0 * alpha; kernel[1][2] = -alpha;
	kernel[2][0] = 0; kernel[2][1] = -alpha; kernel[2][2] = 0;
}

void GetKernel_S2(float alpha, float** kernel) {
	kernel[0][0] = -alpha; kernel[0][1] = -alpha; kernel[0][2] = -alpha;
	kernel[1][0] = -alpha; kernel[1][1] = 1 + 8.0 * alpha; kernel[1][2] = -alpha;
	kernel[2][0] = -alpha; kernel[2][1] = -alpha; kernel[2][2] = -alpha;
}

void ex1106_2() { //선명화 - 2
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

	float alpha = 0.5; //선명화 정도

#if 0
	GetKernel_S1(alpha, kernel);
#else
	GetKernel_S2(alpha, kernel);
#endif

	MaskingImage(height, width, img, kernel, img_out); //마스킹
	ClippingImage(img_out, img_out, height, width); //클리핑

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void ex1106_3() { //선명화 - 3
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(3, 3);

	ImageShow((char*)"input", img, height, width);

	for (float alpha = 0.1; alpha <= 1.0, alpha += 0.1;) {
		GetKernel_S1(alpha, kernel);
		MaskingImage(height, width, img, kernel, img_out); //마스킹
		ClippingImage(img_out, img_out, height, width); //클리핑

		ImageShow((char*)"output", img_out, height, width);
	}
}

void Bubbling(int* data, int size) {
	for (int i = 0; i < size - 1; i++) {
		if (data[i] > data[i + 1]) {
			int temp = data[i];
			data[i] = data[i + 1];
			data[i + 1] = temp;
		}
	}
}

void BubbleSort(int* data, int size) {
	for (int n = 0; n < size - 1; n++) {
		Bubbling(data, size - n);
	}
}

#define SIZE 5
void ex1106_4(){
	int data[SIZE] = { 7,3,2,5,1 };
	/*if (data[0] > data[1]) {
		int temp = data[0];
		data[0] = data[1];
		data[1] = temp;
	}
	if (data[1] > data[2]) {
		int temp = data[1];
		data[1] = data[2];
		data[2] = temp;
	}*/
	/*for (int i = 0; i < 5; i++)
		for (int j = 0; j < 4; j++) {
			if (data[j] > data[j + 1]) {
				int temp = data[j];
				data[j] = data[j + 1];
				data[j + 1] = temp;
			}
		}*/
	//Bubbling(data, SIZE);
	////data[SIZE-1] : 최대값 --> 더 이상 비교할 필요가 없음
	//Bubbling(data, SIZE - 1);
	////data[SIZE-2] : 두 번째로 큰 값 --> 더 이상 비교할 필요가 없음
	//Bubbling(data, SIZE - 2); 
	////data[SIZE-3] : 세 번째로 큰 값 --> 더 이상 비교할 필요가 없음
	//Bubbling(data, SIZE - 3);

	BubbleSort(data, SIZE);
}

void GetBlock3x3(int y, int x, int** img, int* block1D) { //1차원 배열에 3x3 블록 복사
	int index = 0;
	for (int m = -1; m <= 1; m++) {
		for (int n = -1; n <= 1; n++) {
			block1D[index] = img[y + m][x + n];
			index++;
		}
	}
}

void MedianFilter3x3(int** img, int height, int width, int** img_out) {
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int block[9];
			GetBlock3x3(y, x, img, block);
			BubbleSort(block, 9);
			img_out[y][x] = block[4]; //중간값
		}
	}
}

void CopyImage(int** src, int height, int width, int** dst) {
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			dst[y][x] = src[y][x];
		}
}

void ex1112_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lenaSP20.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_org = (int**)IntAlloc2(height, width);

	//MedianFilter3x3(img, height, width, img_out); //중간값 필터
	//
	//MedianFilter3x3(img_out, height, width, img_out2); //중간값 필터를 두 번 적용

	/*ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
	ImageShow((char*)"output2", img_out2, height, width);*/

	//중간값 필터 n회 적용
	//img->img_org copy
	//img_out -> img copy
	CopyImage(img, height, width, img_org);
	for (int n = 0; n < 5; n++) {
		MedianFilter3x3(img, height, width, img_out); //중간값 필터를 다섯 번 적용
		CopyImage(img_out, height, width, img); //img_out -> img copy
		ImageShow((char*)"output", img_out, height, width);
		//너무 많이 돌리면 사진이 뭉개짐
	}

	ImageShow((char*)"input", img_org, height, width);
	//ImageShow((char*)"output", img, height, width);
	//ImageShow((char*)"output2", img_out, height, width);
}

void MaxFilter3x3(int** img, int height, int width, int** img_out) { //팽창	필터(dilate, dilation 필터)
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int block[9];
			GetBlock3x3(y, x, img, block);
			BubbleSort(block, 9);
			img_out[y][x] = block[8]; //최대값
		}
	}
}

void MinFilter3x3(int** img, int height, int width, int** img_out) { //침식 필터(erde, erosion 필터)
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int block[9];
			GetBlock3x3(y, x, img, block);
			BubbleSort(block, 9);
			img_out[y][x] = block[0]; //최대값
		}
	}
}

void ex1112_2() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/bin_numbers.bmp", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_org = (int**)IntAlloc2(height, width);

	CopyImage(img, height, width, img_org); //원본 이미지 저장

	MaxFilter3x3(img, height, width, img_out); //팽창 필터
	CopyImage(img_out, height, width, img); //img_out -> img copy
	MaxFilter3x3(img, height, width, img_out); //팽창 필터 2번 적용

	CopyImage(img_org, height, width, img); //원본 이미지 복사
	MinFilter3x3(img, height, width, img_out2); //침식 필터
	CopyImage(img_out2, height, width, img); //img_out2 -> img copy
	MinFilter3x3(img, height, width, img_out2); //침식 필터 2번 적용

	ImageShow((char*)"input", img_org, height, width);
	ImageShow((char*)"output", img_out, height, width);
	ImageShow((char*)"output2", img_out2, height, width);
}

void ex1112_3() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/bin_numbers.bmp", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_org = (int**)IntAlloc2(height, width);

	CopyImage(img, height, width, img_org); //원본 이미지 저장

	MaxFilter3x3(img, height, width, img_out);
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			img_out2[y][x] = img_out[y][x] - img[y][x]; //팽창 필터 - 원본 이미지
		}

	MinFilter3x3(img, height, width, img_out);

	ImageShow((char*)"input", img_org, height, width);
	ImageShow((char*)"output", img_out, height, width);
	ImageShow((char*)"output2", img_out2, height, width);
}

void UpsamplingX2_0order(int** img, int height, int width, int** img_out) { //0차 보간
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[2 * y][2 * x] = img[y][x]; //단순 확장
			img_out[2 * y][2 * x + 1] = img[y][x];
			img_out[2 * y + 1][2 * x] = img[y][x];
			img_out[2 * y + 1][2 * x + 1] = img[y][x];
		}
	}
}

void UpsamplingX2_1order(int** img, int height, int width, int** img_out) { //1차 보간
	for (int y = 0; y < height; y++) { //(2y,2x) 자리 채운거
		for (int x = 0; x < width; x++) {
			img_out[2 * y][2 * x] = img[y][x]; //짝수자리 채우기
		}
	}
	for (int y = 0; y < height; y++) { //(2y,2x+1)
		for (int x = 0; x < width; x++) {
			img_out[2 * y][2 * x + 1] = (img[y][x] + img[y][x + 1]) / 2.0 + 0.5;
		}
	}
	for (int y = 0; y < height - 1; y++) { //(2y+1,2x)
		for (int x = 0; x < width; x++) {
			img_out[2 * y + 1][2 * x] = (img[y][x] + img[y + 1][x]) / 2.0 + 0.5; //짝수자리 채우기
		}
	}

	int heightx2 = height * 2;
	int widthx2 = width * 2;
	for (int yp = 1; yp < heightx2; yp += 2) { //(2y+1,2x+1)
		for (int xp = 1; xp < widthx2; xp += 2) {
			img_out[yp][xp] = (img_out[yp][xp - 1] + img_out[yp][xp + 1]) / 2.0 + 0.5;
		}
	}
}

void ex1119_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int heightx2 = height * 2;
	int widthx2 = width * 2;
	int** img_out_0 = (int**)IntAlloc2(heightx2, widthx2);
	int** img_out_1 = (int**)IntAlloc2(heightx2, widthx2);

	UpsamplingX2_0order(img, height, width, img_out_0);
	//y축의 위아래 픽셀의 평균으로 빈 자리 채우기
	//x축의 좌우 픽셀의 평균으로 빈 자리 채우기
	
	UpsamplingX2_1order(img, height, width, img_out_1); //가로세로 2배 큰 사진 출력

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output_0", img_out_0, heightx2, widthx2); //좀 자글자글하게 확대됨
	ImageShow((char*)"output_1", img_out_1, heightx2, widthx2); //좀 스무스하게 확대됨
}

void DownSamplingX2(int** img, int height, int width, int** img_out) {
	int half_height = height / 2;
	int half_width = width / 2;

	for (int y = 0; y < height / 2; y++) {
		for (int x = 0; x < width / 2; x++) {
			img_out[y][x] = img[2 * y][2 * x]; //단순 축소
		}
	}
}

void ex1119_2() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int half_height = height / 2;
	int half_width = width / 2;
	int** img_out = (int**)IntAlloc2(half_height, half_width); 
	int** img_out_1= (int**)IntAlloc2(half_height/2, half_width/2);

	DownSamplingX2(img, height, width, img_out);
	DownSamplingX2(img_out, half_height, half_width, img_out_1); //가로세로 1/4배 작은 사진 출력

	ImageShow((char*)"input", img, height, width); 
	ImageShow((char*)"output", img_out, half_height, half_width);
	ImageShow((char*)"output", img_out_1, half_height/2, half_width/2);
}

void ex1119_3() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/zoneplate.png", &height, &width);
	int half_height = height / 2;
	int half_width = width / 2;
	int** img_out = (int**)IntAlloc2(half_height, half_width);
	int** img_out_1 = (int**)IntAlloc2(half_height / 2, half_width / 2);

	int** img_mean = (int**)IntAlloc2(height, width);
	MeanFilter3x3(img, height, width, img_mean);
	DownSamplingX2(img_mean, height, width, img_out);

	int** img_mean_h = (int**)IntAlloc2(half_height, half_width);
	MeanFilter3x3(img_out, half_height, half_width, img_mean_h);
	DownSamplingX2(img_mean_h, half_height, half_width, img_out_1); //가로세로 1/4배 작은 사진 출력

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, half_height, half_width);
	ImageShow((char*)"output", img_out_1, half_height / 2, half_width / 2);
}

int BilinearInterpolation(float y_f, float x_f, int** img, int height, int width) {
	//예외처리 : 좌표 y_f, x_f가 이미지 범위를 벗어나는 경우 0을 리턴
	int y = (int)y_f; //정수형으로 변환
	int x = (int)x_f; //정수형으로 변환

	if (y < 0 || x < 0 || x >= width - 1 || y >= height - 1)
		return 0;
	else {
		int A = img[y][x];
		int B = img[y][x + 1];
		int C = img[y + 1][x];
		int D = img[y + 1][x + 1];
		float dx = x_f - x;
		float dy = y_f - y;

		int value = (1 - dx) * (1 - dy) * A + dx * (1 - dy) * B + (1 - dx) * dy * C + dx * dy * D + 0.5;
		return value;
	}
}

void ex1120_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int heightx2 = height * 2;
	int widthx2 = width * 2;
	int** img_out = (int**)IntAlloc2(heightx2, widthx2);

	//float y_f = 100.5, x_f = 200; //정수형으로 변환해서 소수점 날리기 = 좌표 구하기
	//int value = BilinearInterpolation(y_f, x_f, img, height, width);
	for (int yp = 0; yp < heightx2; yp++) {
		for (int xp = 0; xp < widthx2; xp++) {
			float y = yp / 2.0;  //0.5, 0.7, 1.2,...
			float x = xp / 1.5;	 //0.5, 0.7, 1.2,...
			img_out[yp][xp] = BilinearInterpolation(y, x, img, height, width);
		}
	}
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, heightx2, widthx2);
}

void main() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	double theta = 45.0;
	theta = CV_PI / 180.0 * theta; //degree -> radian

	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			float x = cos(theta) * xp + sin(theta) * yp; //theta : 라디안 값 들어가야 함
			float y = -sin(theta) * xp + cos(theta	) * yp;
			img_out[yp][xp] = BilinearInterpolation(y, x, img, height, width);
		}
	}

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}