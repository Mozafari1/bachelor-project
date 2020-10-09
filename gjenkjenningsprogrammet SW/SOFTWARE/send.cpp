#include "send.h"
send::send() {

	FolderPath = "C:\\OpenCV_DIR\\Gjenkjenning_PVR\\gjenkjenning\\bilder\\";
	imageExtension = ".jpg";
}

send::send(string FolderPath, string imageExtension) {
	FolderPath = FolderPath;
	imageExtension = imageExtension;
}

send::~send(){}
string send::getFolderPath() {
	return FolderPath;


}
string send::getimageExtension() {
	return imageExtension;
}
zbarBarcode::zbarBarcode() {
	
	barcodeType = barcodeType;
	barcodeData = barcodeData;
}
zbarBarcode::zbarBarcode( string barcodeType, string barcodeData) {
	
	barcodeType = barcodeType;
	barcodeData = barcodeData;
}
zbarBarcode::~zbarBarcode() {

}

