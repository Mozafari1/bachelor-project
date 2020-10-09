#include <iostream>
#include<string>
#include <vector>
#include<opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core.hpp"
#include <float.h>
#include "mottaPartA.h"
#include"mottaPartB.h"
#include"send.h";
#include "zbar.h";
#include<list>
#include<fstream>
#include <filesystem>
#include<boost/filesystem.hpp>
#include<algorithm> 
int mottaPA();
int mottaPB();
int test();
string readAndDecodeBarcode(Mat& frame);
int sende();
int filetest();
int mpb();
int tst();
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace zbar;
namespace fs = experimental::filesystem;

int main(int argc, char* argv[]) {
	// These function must connects to  buttons in User Interface.
	// When user press a button one of the functions calls

	//mottaPA();      // -- This function calls when the user press the button to take picture for recognation
	//mottaPB();		 // -- This one calls when the user wants to compare one image with other images. The user can press this button again if the image didnt get match.
	//test();
	//sende();		//-- This function calls when the user wants to read barcode and take pictures
	//filetest();
	//mpb();
	tst();
	return 0;
}


// File testing


int filetest() {
	boost::filesystem::path p("C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bilder/");
	for (auto i = boost::filesystem::directory_iterator(p); i != boost::filesystem::directory_iterator(); i++)
	{
		if (!is_directory(i->path())) //we eliminate directories in a list
		{
			cout << i->path().filename().string() << endl;
		}
		else
			continue;
	}
	return 0;
}








// Ending file testing
int test() {
	vector<String > fn;
	vector<Mat> images;
	String imagename = "*.jpg";
	String Path = "C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bilder/";
	glob(Path + imagename, fn, true);

	size_t count = fn.size(); //number of png files in images folder

	for (size_t i = 0; i < count; ++i) {

		Mat im = imread(fn[i]);
		if (!im.empty()) {
			//continue;
			cout << i << endl;
		}


		images.push_back(im);

	}


	return 0;
}

// The mottaPA() function is create to capture video frame from the webcamera and blurring the image.
// This function showing the frame on the screen and saving the image in a folder with PATH (Check getfullPATH() function in mottaPartA.cpp)
// If the camera is not open or the PORT is not defined correct it will cast and error message.
// The number (1) is defined for camera wich is conneted through USB.
// When the user press close button on the screen the program will close.

int  mottaPA() {
	mottaPartA mpa;

	VideoCapture capture(1);
	if (!capture.isOpened()) {
		cout << " Fant ingen frame, sjekk om webkamera er koblet" << endl;
		return false;
	}
	else {
		while (true)
		{
			Mat frame, blur;

			capture >> frame;
			GaussianBlur(frame, blur, Size(5, 5), 0);
			imshow("Webkamera blurred", blur);
			//			ESC erstattes med en knap
			if (waitKey(30) == 27) { // esc							
				imwrite(mpa.getfullPATH(), blur);
				cout << mpa.getimageName() << " ble lagret" << endl;
			}
			// erstattes med en knap
			if (waitKey(50) == 113) { // q
				break;
			}
		}
	}
	return 0;
}


//The function mottaPB is to compare images and show the results to screen and printing out the match percent. 
int mottaPB() {


	mottaPartB mpb;
	string FILEPATH = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\EAN\\percent.txt";
	mpb.detector->detectAndCompute(mpb.getImage(), noArray(), mpb.keypoint_1, mpb.descriptor_1);										// Detecting keypoints_1 and computes the descriptor_1

	mpb.detector->detectAndCompute(mpb.getImages(), noArray(), mpb.keypoint_2, mpb.descriptor_2);										// Detecting keypoints_2 and computes the descriptor_2


	mpb.matcher->knnMatch(mpb.descriptor_1, mpb.descriptor_2, mpb.k_nearest_neighbor_matches, 2);										//  Matching descriptor and finding the best K match


	for (size_t i = 0; i < mpb.k_nearest_neighbor_matches.size(); i++)																	// Filter matches. Basically this loop filtering the matches and
	{
		if (mpb.k_nearest_neighbor_matches[i][0].distance < mpb.ratio_threshhold * mpb.k_nearest_neighbor_matches[i][1].distance)
		{
			mpb.good_match_points.push_back(mpb.k_nearest_neighbor_matches[i][0]);
		}
	}
	// Printing out the match percent
	int num_keypoints = 0;
	if (mpb.keypoint_1.size() > mpb.keypoint_2.size()) {

		num_keypoints = mpb.keypoint_1.size();
	}
	else {
		num_keypoints = mpb.keypoint_2.size();
	}
	int per = mpb.good_match_points.size() * 100 / num_keypoints;
	ofstream EAN_FILE(FILEPATH);
	if (EAN_FILE.is_open()) {
		EAN_FILE << per << endl;
		EAN_FILE.close();
	}
	else {
		cout << "kunne ikke lagre prosent";
	}
	cout << "Match percent " << per << endl;
	cout << "Image name " << mpb.getImageName() << endl;

	drawMatches(mpb.getImage(), mpb.keypoint_1, mpb.getImages(), mpb.keypoint_2, mpb.good_match_points, mpb.draw_matches, Scalar::all(-1),	//-- Drawing matches, but not single keypoints
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show  matches
	imwrite("C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\result\\result.jpg", mpb.draw_matches);

	imshow("Results", mpb.draw_matches);
	waitKey(30);
	return 0;

}




// This function is create to decode barcode.
// In this function I use classes from ZBar Library and from OpenCV
// This fucntion is called by the sende() function.

string readAndDecodeBarcode(Mat& frame) {
	zbarBarcode decodeBarcode;

	ImageScanner s;																					// ImageScanner is a class of Zbar using this to scan the image
	s.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);													// Enabling for barcode 
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);															// We have to convert the frame into gray other ways it the program cant decode the barcode
	Image img(frame.cols, frame.rows, "Y800", (uchar*)gray.data, frame.cols * frame.rows);			// Using Image class from Zbar to get data from cols and rows.

	int x = s.scan(img);																			// Scanning and returning data
	string barcodeData;

	for (Image::SymbolIterator i = img.symbol_begin(); i != img.symbol_end(); ++i) {				// This loops itererets over the barcode data and get type and data

		decodeBarcode.barcodeType = i->get_type_name();
		decodeBarcode.barcodeData = i->get_data();


		cout << "Data : " << decodeBarcode.barcodeData << endl;
		//cout << "Type : " << decode.barcodeType  << endl;

		barcodeData = decodeBarcode.barcodeData;
	}
	return barcodeData;

}
// This function reads frame from kamera and takes pictures and reads the barcode. When the barcode is readed it will be stored temporary in a text file.
// When the picture is taken it will get the barcode data as its name. The file text file is overwriting.
// In this function I'm using basic method to open, save and read to and from file.
int sende() {


	send send;
	string EAN;
	string LINE;
	string FILEPATH = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\EAN\\ean.txt";
	VideoCapture capture(1);
	if (!capture.isOpened()) {
		cout << " Fant ingen frame, sjekk om webkamera er koblet" << endl;
		return false;
	}
	else {
		while (true)
		{
			Mat frame, blur;

			capture >> frame;

			GaussianBlur(frame, blur, Size(5, 5), 0);
			EAN = readAndDecodeBarcode(frame);

			if (EAN.empty()) {
				cout << "Empty EAN" << endl;
			}
			else {
				ofstream EAN_FILE(FILEPATH);

				if (EAN_FILE.is_open()) {
					EAN_FILE << EAN << endl;
					EAN_FILE.close();
				}
				else {
					cout << "EAN filen er utilgjengelig for å skrive barkoden";
				}
			}
			ifstream EAN_FILE(FILEPATH);
			string BARCODE = LINE;
			if (EAN_FILE.is_open()) {
				getline(EAN_FILE, LINE);
				EAN_FILE.close();
			}

			imshow("Webkamera blurred", blur);
			string fullPath = send.getFolderPath() + BARCODE + send.getimageExtension();

			//			ESC erstattes med en knap
			if (waitKey(30) == 27) { // esc
				imwrite(fullPath, blur);
				cout << fullPath << " ble lagret" << endl;
			}
			// erstattes med en knap
			if (waitKey(50) == 113) { // q
				break;
			}
		}
	}
	return 0;
}



// Part b

int mpb() {
	boost::filesystem::path p("C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bilder/");
	vector	<String> fn;
	vector <Mat> img;
	String imageEx = "*.jpg";
	String Path = "C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bilder/";
	String fp = Path + imageEx;
	glob(fp, fn);
	size_t count = fn.size();

	string FP = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\EAN\\name.txt";
	mottaPartB mpb;
	string FILEPATH = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\EAN\\percent.txt";
	mpb.detector->detectAndCompute(mpb.getImage(), noArray(), mpb.keypoint_1, mpb.descriptor_1);										// Detecting keypoints_1 and computes the descriptor_1
	for (size_t x = 0; x < count; x++)
	{

		Mat im = imread(fn[x]);
		//img.push_back(im);


		//cout << "this is im: " << im << endl;

	//}
	//for (vector<Mat>::iterator itr = img.begin(); itr != img.end(); itr++) {

		mpb.detector->detectAndCompute(im, noArray(), mpb.keypoint_2, mpb.descriptor_2);
		mpb.matcher->knnMatch(mpb.descriptor_1, mpb.descriptor_2, mpb.k_nearest_neighbor_matches, 2);

		for (size_t i = 0; i < mpb.k_nearest_neighbor_matches.size(); i++)																	// Filter matches. Basically this loop filtering the matches and
		{
			if (mpb.k_nearest_neighbor_matches[i][0].distance < mpb.ratio_threshhold * mpb.k_nearest_neighbor_matches[i][1].distance)
			{
				mpb.good_match_points.push_back(mpb.k_nearest_neighbor_matches[i][0]);
			}
		}

		int num_keypoints = 0;
		if (mpb.keypoint_1.size() > mpb.keypoint_2.size()) {

			num_keypoints = mpb.keypoint_1.size();
		}
		else {
			num_keypoints = mpb.keypoint_2.size();
		}
		int per = mpb.good_match_points.size() * 100 / num_keypoints;
		ofstream EAN_FILE(FILEPATH, ofstream::app);

		if (EAN_FILE.is_open()) {
			EAN_FILE << per << endl;

			EAN_FILE.close();
		}
		else {
			cout << "kunne ikke lagre prosent";
		}
		cout << "Match percent " << per << endl;
		cout << "Image gm " << mpb.good_match_points.size() << endl;
		cout << "Image k1 " << mpb.keypoint_1.size() << endl;
		cout << "Image k2 " << mpb.keypoint_2.size() << endl;
		//cout << "image matrix " << im << endl;
		ofstream EF(FP, ofstream::app);

		if (EF.is_open()) {
			EF << im << endl;
			EF.close();
		}
		else {
			cout << "kunne ikke lagre navn" << endl;
		}
		//drawMatches(mpb.getImage(), mpb.keypoint_1, im, mpb.keypoint_2, mpb.good_match_points, mpb.draw_matches, Scalar::all(-1),	//-- Drawing matches, but not single keypoints
		//	Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//
		//-- Show  matches
		//imwrite("C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\result\\result.jpg", mpb.draw_matches);

		//imshow("Results", mpb.draw_matches);


	}

	waitKey(30);

	return 0;

}

int tst() {
	string Line;
	vector	<String> fn;
	vector <Mat> img;
	String imageEx = "*.jpg";
	String Path = "C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bilder/";
	String fp = Path + imageEx;
	glob(fp, fn);
	size_t count = fn.size();
	String Path1 = "C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bildet/A4 PC-0004 A.jpg";
	String FILEPATH = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\EAN\\percent.txt";
	//String Path2 = "C:/OpenCV_DIR/Gjenkjenning_PVR/gjenkjenning/bildet/A4 PC-0004 A.jpg";
	String FP = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\EAN\\name.txt";

	Mat img1 = imread(Path1, IMREAD_GRAYSCALE);
	//Mat img2 = imread(Path2, IMREAD_GRAYSCALE);

	Ptr<SURF> detector = SURF::create(400);
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	for (size_t x = 0; x < count; x++)
	{

		Mat img2 = imread(fn[x]);

		detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

		const float ratio_thresh = 0.7f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}
		int num_keypoints = 0;
		if (keypoints1.size() > keypoints2.size()) {

			num_keypoints = keypoints1.size();
		}
		else {
			num_keypoints = keypoints2.size();
		}
		int per = good_matches.size() * 100 / num_keypoints;
		if (per >60) {
			ofstream EAN_FILE(FILEPATH, ofstream::app);

			if (EAN_FILE.is_open()) {
				EAN_FILE << per << endl;
				EAN_FILE.close();
			}
			else {
				cout << "kunne ikke lagre prosent";
			}
			ofstream EN(FP, ofstream::app);

			if (EN.is_open()) {
				EN << fn[x] << endl;

				EN.close();
			}
			else {
				cout << "kunne ikke lagre prosent";
			}
			cout << "Good  Match " << per << endl;
			cout << "Image Path" <<fn[x] << endl;
			Mat img_matches;
			drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
				Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			
			imwrite("C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\result\\result.jpg",  img_matches);
			
		}
		else {
			cout << "Bad Match percent: " << per << endl;
			cout << "Image Path" << fn[x] << endl;
		}
	

		
		cout << "Good matches " << good_matches.size() << endl;
		cout << "Image k1 " << keypoints1.size() << endl;
		cout << "Image k2 " << keypoints2.size() << endl;
		cout << "----------------------"<< endl;
		//Mat img_matches;
		//drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
		//	Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		////-- Show detected matches
		//cout <<"Img matches: " <<img_matches << endl;

	}
	Mat final_img = imread("C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\result\\result.jpg");
	imshow("Final result", final_img);
	
	
	waitKey(10000);
	return 0;


}

