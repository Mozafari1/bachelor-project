#include "gjenkjenning_send.h"
#include "ui_gjenkjenning_send.h"


#include<QDebug>
#include <QtCore>
#include<QMessageBox>
#include<fstream>
#include<iostream>
#include <sstream>
#include<string>
#include<QZXing.h>
#include<QImageReader>
#include <QThread>
// Constructor /////////////////////////////////////////////////////


gjenkjenning_send::gjenkjenning_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::gjenkjenning_send)
{
    ui->setupUi(this);
    this->setWindowState(Qt::WindowMaximized);

    this->setWindowTitle("Parts Vision Recognition");
    ///////////////////////////////////////////////////


}
// DeConstructor /////////////////////////////////////////////////////
gjenkjenning_send::~gjenkjenning_send()
{
    delete ui;
}

/////////////////////////////////////////////////////////////// EXIT PROGRAM

void gjenkjenning_send::exitProgram(){
    if(qtimer->isActive()) qtimer->stop();          // if timer is running, stop timer
    QApplication::quit();                           // and exit the program
}

///////////////////////////////////////////////////////////////////////////
//              processFrameAndUpdateGUI()
///////////////////////////////////////////////////////////////////////////

void gjenkjenning_send::processFrameAndUpdateGUI(){
    cv::Mat imgOriginal;            // Input image

    //capWebcam.read(imgOriginal);

    bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);    // Get next frame from the webcam

    if (!blnFrameReadSuccessfully|| imgOriginal.empty()){                                                // if we did not get a frame
        QMessageBox::information(this, "", "Unable to read from webcam \n\n exiting program\n");        // SHOW ERROR MSG
        exitProgram();      // and exit program
        return;
    }

    QImage qimageOriginal = convertOpenCVMatToQImage(imgOriginal);           // Convert from to OpenCV to Qt QImage
    ui->labelFrame->setPixmap(QPixmap::fromImage(qimageOriginal));       // Show images on form labels


}


////////////////////////////////////////////////////////////////////////
/////////// FUNCTIONS THAT CONVERTING FROM OPENCV TO Qt QImage
////////////////////////////////////////////////////////////////////////

QImage gjenkjenning_send::convertOpenCVMatToQImage(cv::Mat mat){
    if(mat.channels()==1){          // if grayscale image
        return QImage((uchar*) mat.data, mat.cols, mat.rows,mat.step, QImage::Format_Indexed8);         // Declare and return a Qimage
    }else if(mat.channels()==3){            // if 3 channel color image
        //cv::cvtColor(mat, mat, CV_BGR2RGB);        // Invert BRG to RGB
        return  QImage((uchar*) mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888);          // Declare and reeturn a QImage

    }else{
        qDebug()<<"In convertOpenCVMatToQImage, image was not 1 channel or 3 channel, should never get here";
    }
    return QImage();
}



void gjenkjenning_send::on_pushButtonScanBarcode_clicked()
{
    cv::Mat imgOriginal;

      bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);    // Get next frame from the webcam
      if (!blnFrameReadSuccessfully|| imgOriginal.empty()){                                                // if we did not get a frame
          QMessageBox::information(this, "", "Unable to read from webcam \n\n exiting program\n");        // SHOW ERROR MSG
          exitProgram();      // and exit program
          return;
      }

      QImage qimageOriginal = convertOpenCVMatToQImage(imgOriginal);
      QString filename = "barcode.png";
      qimageOriginal.save("C:/OpenCV_DIR/Qt/PVR/EAN/"+filename,"PNG");
      encodeBarcode();
}

///////////////////////////////////////////////////////////////////
//                  Taking picture
///////////////////////////////////////////////////////////////////

void gjenkjenning_send::on_pushButtonTakePicture_clicked()

{   cv::String Path = "C:/OpenCV_DIR/Qt/PVR/EAN/";

    cv::Mat imgOriginal;


    cv::Mat barcodeRead = cv::imread(Path+"barcode.png");
    QImage qimageBarcode = convertOpenCVMatToQImage(barcodeRead);                  // Convert from to OpenCV to Qt QImage
    QZXing decoder;
    decoder.setDecoder( QZXing::DecoderFormat_QR_CODE | QZXing::DecoderFormat_CODE_39 |QZXing::DecoderFormat_EAN_8 | QZXing::DecoderFormat_EAN_13
                        | QZXing::DecoderFormat_CODE_93 | QZXing::DecoderFormat_CODE_128 | QZXing::DecoderFormat_PDF_417);

    QString result = decoder.decodeImage(qimageBarcode);

      bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);    // Get next frame from the webcam
      if (!blnFrameReadSuccessfully|| imgOriginal.empty()){                                                // if we did not get a frame
          QMessageBox::information(this, "", "Unable to read from webcam \n\n exiting program\n");        // SHOW ERROR MSG
          exitProgram();      // and exit program
          return;
      }
      QString barcodeData = result;
      QImage qimageOriginal = convertOpenCVMatToQImage(imgOriginal);
      QString ImageExtention = ".jpg";

      QString filename = barcodeData+ImageExtention;
      //qInfo() <<"this is barcode "<<barcodeData;
      if(filename==".jpg"){
           ui->labelSavingImage->setText("Bildet ble ikke lagret");
           ui->labelBarcodeData->setText("Ingen barkode. Vennligst les barkoden på nytt");
      }else{
          qimageOriginal.save("C:/OpenCV_DIR/Qt/PVR/images/"+filename,"JPG");
          ui->labelBarcodeData->setText(barcodeData);
          ui->labelSavingImage->setText("Bildet ble lagret");
      }

}

///////////////////////////////////////////////////////////////
// Close the program
//////////////////////////////////////////////////////////////
void gjenkjenning_send::on_pushButtonClose_clicked()
{
    exitProgram();
}


void gjenkjenning_send::on_pushButtonStartKamera_clicked()
{
    capWebcam.open(1);                          // Assocciate the capture object to the default camera
    if(capWebcam.isOpened()==false){                                                                               // if unsuccessful
        QMessageBox::information(this, "", "Error: capWebcam not accessed successfully \n\n exiting program \n"); //SHOW  Error msg
        exitProgram();
        return;
    }
    qtimer = new QTimer(this);                                                                 // Instantiate timer
    connect(qtimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));                 // Associate timer to processFrameAndUpdateGUI
    //qtimer->singleShot(50000, this, SLOT(processFrameAndUpdateGUI()));

    qtimer->start(20);


}

int gjenkjenning_send::encodeBarcode(){

        QString data = "RA-234";
        QImage barcode = QZXing::encodeData(data);

        //qInfo() <<"This is barcodede"<< barcode;
        return 1;
}
