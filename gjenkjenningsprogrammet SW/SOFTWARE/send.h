#pragma once
#include <iostream>
#include<opencv2/opencv.hpp>
#include<string>
#include"zbar.h"
#include<vector>

#ifdef send
#define send
#endif // send
using namespace std;
using namespace cv;
using namespace zbar;


 
class send
{
public:
	send();							// default constructor, returning default values 
	send( string, string);			// Overload constructor
	~send();						// deconstructor
	
	string getFolderPath();			// These two functions are defined to access to private variables
	string getimageExtension();
private:
									// They are just private variables ogf type string
	string FolderPath;
	string imageExtension;

};

// This is class  is create for Zbar barcode
// This is a very simple class where it just contains constructor and two public variables
class zbarBarcode {
public:
	zbarBarcode();						
	zbarBarcode(  string, string);
	~zbarBarcode();


	string barcodeType;
	string barcodeData;
private:
	
};