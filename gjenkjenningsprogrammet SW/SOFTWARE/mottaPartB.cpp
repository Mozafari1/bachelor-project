#include "mottaPartB.h"

// default constructor
// Return default value
mottaPartB::mottaPartB() {
	PATH = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\bildet\\"; 
	imageName = "A4 PC-0004 A.jpg";
	fullPATH = PATH + imageName;

	image = imread(fullPATH,IMREAD_GRAYSCALE);										// Reading image the first image
	//-----------------------------------------------------
	PATH_2 = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\bilder\\";
	imageName_2 = "DR2039.jpg";
	fullPATH_2 = PATH_2 + imageName_2;
	images = imread(fullPATH_2, IMREAD_GRAYSCALE);									// Reading the second image to compare with the first image

	//------------ Ikke ferdig. Må finne en måte å loope over alle bilder.  

	//filename = filename;
	//
	//glob(fullPATH_2, filename, false);
	//for (size_t i = 0; i <= filename.size(); i++) {
	//	images.push_back(imread(filename[i]));
	//}

	//----------------------------------------



	detector =xfeatures2d::SURF::create(1900);										// Creating SURF feature and using hessian threshhold. Threshhold value can changes 
	descriptor_1 = descriptor_1;													// Gets default value
	descriptor_2 = descriptor_2;
	keypoint_1 = keypoint_1;
	keypoint_2 = keypoint_2;
	
	matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);				// Creating a Flann based descriptor of keypoints 
	k_nearest_neighbor_matches = k_nearest_neighbor_matches;						// Again, gets default value
	good_match_points = good_match_points;
	draw_matches = draw_matches;
	

}

// Overloading constructor
// Return some value of data type string and Mat
mottaPartB::mottaPartB(Mat image, Mat images,  string fullPATH,
	string PATH , string imageName, string fullPATH_2, string PATH_2, string imageName_2) {
	image = image;
	images = images;

	fullPATH = fullPATH;
	PATH = PATH;
	imageName = imageName;
	
	fullPATH_2 = fullPATH_2;
	PATH_2 = PATH_2;
	imageName_2 = imageName_2;

}


mottaPartB::mottaPartB::~mottaPartB() {
	// Deconstructor
}

// These functions just return some value. They are needed to get access to private variables

Mat mottaPartB::getImage() const {
	return image;
}
Mat mottaPartB::getImages()const {
	return images;
}

string mottaPartB::getFullPATH() const {
	return fullPATH;
}
string mottaPartB::getPATH() const {
	return	PATH;
}
string mottaPartB::getImageName()const {
	return imageName;
}

string mottaPartB::getFullPATH_2() const {
	return fullPATH;
}
string mottaPartB::getPATH_2() const {
	return	PATH;
}
string mottaPartB::getImageName_2()const {
	return imageName;
}
