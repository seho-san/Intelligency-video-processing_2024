//2020037049 김세호
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>

using namespace cv; //cv::쓸 때 cv:: 안써도 됨

typedef struct {
	int r, g, b; //.
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
	Mat img=imread("anq.png");

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

void ex0925_1() {
	 int A = 100, B = 200, C = 300;
	 /*int D = GetMax(A, B);
	 int E = GetMax(D, C);*/

	 int E = GetMax(GetMax(A, B), C); //최댓값 구하는 매크로를 사용해 한 줄로 표현
	 //중간값 구하는 매크로도 만들 수 있음
	 //GetMid(x, y, z) ((x > y) ? ((y > z) ? y : ((x > z) ? z : x)) : ((x > z) ? x : ((y > z) ? z : y)));
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
				img_out[y][x] = (255.0 / a) * img[y][x] + 0.5;//255.0은 실수형으로 나누기 위함
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
				//(float)(255.0 / a) 이런식으로 하면 정수나온다 나온답에 실수를 붙이는 거기 때문에. 안하는   게 좋다.   
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
				//(float)(255.0 / a) 이런식으로 하면 정수나온다 나온답에 실수를 붙이는 거기 때문에. 안하는   게 좋다.   
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
				//(float)(255.0 / a) 이런식으로 하면 정수나온다 나온답에 실수를 붙이는 거기 때문에. 안하는   게 좋다.   
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

	chist[0] = histogram[0];
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
	GetHistogram2(height, width, hist_output, img_out);//output 히스토그램 생성

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