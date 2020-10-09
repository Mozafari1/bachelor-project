#pragma once
#include <iostream>
#include<opencv2/opencv.hpp>
#include<string>

#ifdef mottaPartA
#define mottaPartA
#endif // mottaPartA


using namespace std;
using namespace cv;


class mottaPartA
{
public:
	mottaPartA();							// Defualt constructor
	mottaPartA(string, string, string);		// Overload constructor
	~mottaPartA();							// Deconstructor
			
	string getPATH() const;					// Defined functions to get access to private variables 
	string getimageName() const;
	string getfullPATH() const;

private:
											// String variables
	String PATH;
	String imageName;
	String fullPATH;
	

};

