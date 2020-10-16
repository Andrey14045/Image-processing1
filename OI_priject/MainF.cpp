#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <ctime>

using namespace cv;
using namespace std;

double PSNR(Mat& img_orig, Mat& img_com)
{
	Mat s1;
	absdiff(img_orig, img_com, s1);
	s1.convertTo(s1, CV_32F);
	s1 = s1.mul(s1);

	Scalar s = sum(s1);

	double sse = s.val[0] + s.val[1] + s.val[2];

	if (sse <= 1e-10)
		return 0;
	else
	{
		double mse = sse / (double)(img_orig.channels() * img_orig.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}
double compare(double val)
{
	if (val < 0.0)
		return 0.0;
	if (val > 255.0)
		return 255.0;
	return val;
}
void GrayWorld(Mat& RGB, Mat& GRAY)
{
	int R, G, B, Gray;
	GRAY = RGB.clone();
	for (int i = 0; i < RGB.cols; i++)
		for (int j = 0; j < RGB.rows; j++)
		{
			R = RGB.at<cv::Vec3b>(i, j)[2];
			G = RGB.at<cv::Vec3b>(i, j)[1];
			B = RGB.at<cv::Vec3b>(i, j)[0];
			Gray = compare(0.2952*R + 0.5547*G + 0.148*B);
			GRAY.at<cv::Vec3b>(i, j)[0] = Gray;
			GRAY.at<cv::Vec3b>(i, j)[1] = Gray;
			GRAY.at<cv::Vec3b>(i, j)[2] = Gray;
		}
}
void RGB2HSV(Mat& HSV, Mat& RGB)
{
	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols; j++)
		{
		 
			double r = RGB.at<cv::Vec3b>(i, j)[2]/255.0; double g = RGB.at<cv::Vec3b>(i, j)[1]/255.0; double b = RGB.at<cv::Vec3b>(i, j)[0]/255.0;
			double h = 0.0;
			double s = 0.0;
			double v = 0.0;
			double min1 = min(min(r, g), b);
			double max1 = max(max(r, g), b);
			v = max1;               // v

			double delta = max1 - min1;

			if (max1 != 0.0)
				s = 1 - (min1 / max1);       // s
			else {
				// r = g = b = 0       // s = 0, v is undefined
				s = 0.0;
				h = -1.0;
			}
			if (s == 0.0)
				h = 0.0;
			if (r == max1)
				if(g<b)
					 h = 60.0 *(g - b) / delta  +360.0;     // between yellow & magenta
				else h=60.0* (g - b) / delta;
			else if (g == max1)
				h = 60.0*(g-b) / delta+120.0;   // between cyan & yellow
			else
				if(b==v)
				h = 60.0*(g-b) / delta + 240.0;   // between magenta & cyan

			if (h < 0.0)
				h += 360.0;
			HSV.at<cv::Vec3b>(i, j)[0] = compare(h / 360 * 255);
			HSV.at<cv::Vec3b>(i, j)[1] = compare(s * 255.0);
			HSV.at<cv::Vec3b>(i, j)[2] = compare(v * 255.0);
		
		}
	
	}

}



void HSVtoRGB(Mat& HSV, Mat& RGB) {
	for (int i = 0; i < HSV.rows; i++)
	{
		for (int j = 0; j < HSV.cols; j++)
		{
			float H = HSV.at<cv::Vec3b>(i, j)[0]; float S = HSV.at<cv::Vec3b>(i, j)[1]; float V = HSV.at<cv::Vec3b>(i, j)[2];


			if (H > 360 || H < 0 || S>100 || S < 0 || V>100 || V < 0) {
				cout << "The givem HSV values are not in valid range" << endl;
				return;
			}
			float s = S / 100;
			float v = V / 100;
			float C = s * v;
			float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
			float m = v - C;
			float r = 0.0; float g = 0.0;float b = 0.0;
			if (H >= 0 && H < 60) {
				r = C, g = X, b = 0;
			}
			else if (H >= 60 && H < 120) {
				r = X, g = C, b = 0;
			}
			else if (H >= 120 && H < 180) {
				r = 0, g = C, b = X;
			}
			else if (H >= 180 && H < 240) {
				r = 0, g = X, b = C;
			}
			else if (H >= 240 && H < 300) {
				r = X, g = 0, b = C;
			}
			else {
				r = C, g = 0, b = X;
			}
			/*int R = (r + m) * 255;
			int G = (g + m) * 255;
			int B = (b + m) * 255;*/
			int R = (r + m);
			int G = (g + m);
			int B = (b + m);
			RGB.at<cv::Vec3b>(i, j)[0] = compare(R);
			RGB.at<cv::Vec3b>(i, j)[1] = compare(G);
			RGB.at<cv::Vec3b>(i, j)[2] = compare(B);
		}
	}
}
void higher_bright(Mat& img)
{

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			for (int r = 0; r < img.channels(); r++)
				img.at<cv::Vec3b>(i, j)[r] = compare(img.at<cv::Vec3b>(i, j)[r] + 50.0); 
}
int main() 
{   
	Mat img = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_COLOR);
	imshow("Elena", img);
	// #1
	Mat img1, img2;
	img1 = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original image", img1);
	img2 = imread("C:/Users/днс/source/repos/OI_priject/photos/2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Compressed  image", img2);
	double psnr = PSNR(img1, img2);
	cout << "My_PSNR : " << psnr << endl;

	// #2
	Mat img3 = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat img4 = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat gray_img3,gray_img4;
	unsigned int start_my = clock();
	GrayWorld(img3, gray_img3);
	unsigned int end_my = clock();
	unsigned int time_my = end_my - start_my;
	imshow("My_Gray image", gray_img3);
	cout << "My GrayWorld: " << time_my << endl;
	unsigned int start_CV = clock();
	cvtColor(img4, gray_img4, CV_BGR2GRAY);
	unsigned int end_CV = clock();
	unsigned int time_CV = end_CV - start_CV;
	imshow("Gray_CV image ", gray_img4);
	cout << "CV GrayWorld: " << time_CV << "   We are losers :(" << endl ;

	// #3
	Mat myrgb = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat HSV = myrgb.clone();

	unsigned int start_MY = clock();
	RGB2HSV(HSV,myrgb );
	unsigned int end_MY = clock();
	unsigned int time_MY = end_MY - start_MY;
	imshow("RGB->HSV image", HSV);
	cout << "My_HSV: " << time_MY << endl;
	Mat img5 = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat tmp;
	unsigned int start_cv = clock();
	cvtColor(img5, tmp, CV_BGR2HSV);
	unsigned int end_cv = clock();
	unsigned int time_cv = end_cv - start_cv;
	imshow("RGB->HSV OPENCV", tmp);
	cout << "CV_HSV: " << time_cv << endl;
	///////////////////////////////////////////
	Mat my_bright = img5.clone();
	unsigned int start_My = clock();
	higher_bright(my_bright);
	unsigned int end_My = clock();
	unsigned int time_My = end_My - start_My;
	imshow("My bright", my_bright);
	cout << "My bright: " << time_My << endl;

	Mat img6 = imread("C:/Users/днс/source/repos/OI_priject/photos/1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat img_bright;
	unsigned int start_Cv = clock();
	img6.convertTo(img_bright, -1, 1, 50);
	unsigned int end_Cv = clock();
	unsigned int time_Cv = end_Cv - start_Cv;
	imshow("СV bright: ", img_bright);
	cout << "CV bright: " << time_Cv << endl;

	double filter_quality = PSNR(my_bright, img_bright);
	cout << "PSNR filter quality: " << filter_quality << endl;

	waitKey(0);
	return 0;
}