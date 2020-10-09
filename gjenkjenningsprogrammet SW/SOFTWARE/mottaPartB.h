#pragma once
#include <iostream>
#include <vector>
#include <string>
#include<float.h>
#include<map>
#include "opencv2/core.hpp"
#include <list>

#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class mottaPartB
{
public:
	// --Defining public variables
	//-- Variables of cv::class

	Ptr<xfeatures2d::SURF> detector;						// Exctracting SURF feature from image
	vector<KeyPoint> keypoint_1, keypoint_2;				// Storing keypoints 
	Ptr<DescriptorMatcher> matcher;							// Matching descriptor
	vector<vector<DMatch>> k_nearest_neighbor_matches;		// Matching keypoints descriptor
	vector<DMatch> good_match_points;						
	const float  ratio_threshhold =0.9f;
	Mat descriptor_1;										// Mat is matrix which has two data, address where data is stored and the pixel values
	Mat  descriptor_2;
	Mat draw_matches;
	vector<String> filename;

	// Function are defined here, but the body function are created in mottaPartB.cpp
	
	mottaPartB();															 // --default constructor
	mottaPartB( Mat, Mat,  string, string, string, string, string, string);  // This constructor return some data types
	~mottaPartB();                                                           // --Deconstructor

	// Defining functions to access data
	Mat getImage() const;
	Mat getImages() const;
	string getFullPATH() const;
	string getPATH() const ;
	string getImageName()const ;
	string getFullPATH_2() const;
	string getPATH_2() const;
	string getImageName_2()const;




private:
	// -- Defining members variables
	//-- Variable of cv::class 

	Mat image;
	Mat images;
	
	// -- Variables of string type
	
	string fullPATH;
	string PATH;
	string imageName;
	string fullPATH_2;
	string PATH_2;
	string imageName_2;
	

};


