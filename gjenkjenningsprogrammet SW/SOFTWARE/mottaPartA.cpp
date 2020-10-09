#include "mottaPartA.h"

// Default constructor. Returning default values
mottaPartA::mottaPartA() {
	PATH = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\bildet\\";
	imageName = "A4 C-0000 A.jpg";
	fullPATH = PATH + imageName;

}

// Overload constructor. returing some data
mottaPartA::mottaPartA(string PATH, string imageName, string fullPATH) {
	PATH = PATH;
	imageName = imageName;
	fullPATH = PATH+ imageName;

}

// Deconstructor 
mottaPartA::~mottaPartA() {
	// can clear the memory
}

// These functions are created to get access to private variables
string mottaPartA::getPATH()const {
	return PATH;
}
string mottaPartA::getimageName() const {
	return imageName;
}

string mottaPartA::getfullPATH() const {
	return fullPATH;
}
