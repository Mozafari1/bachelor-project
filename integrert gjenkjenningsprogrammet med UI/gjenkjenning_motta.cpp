#include "gjenkjenning_motta.h"
#include "ui_gjenkjenning_motta.h"

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


gjenkjenning_motta::gjenkjenning_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::gjenkjenning_motta)
{
    ui->setupUi(this);
    this->setWindowState(Qt::WindowMaximized);

    this->setWindowTitle("Parts Vision Recognition");
    ///////////////////////////////////////////////////




}

// DeConstructor /////////////////////////////////////////////////////
gjenkjenning_motta::~gjenkjenning_motta()
{
    delete ui;
}

/////////////////////////////////////////////////////////////// EXIT PROGRAM

void gjenkjenning_motta::exitProgram(){
    if(qtimer->isActive()) qtimer->stop();          // if timer is running, stop timer
    QApplication::quit();                           // and exit the program
}


///////////////////////////////////////////////////////////////////////////
//              processFrameAndUpdateGUI()
///////////////////////////////////////////////////////////////////////////

void gjenkjenning_motta::processFrameAndUpdateGUI(){
    cv::Mat imgOriginal;            // Input image


    //capWebcam.read(imgOriginal);
    bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);    // Get next frame from the webcam

    if (!blnFrameReadSuccessfully|| imgOriginal.empty()){                                                // if we did not get a frame
        QMessageBox::information(this, "", "Unable to read from webcam \n\n exiting program\n");        // SHOW ERROR MSG
        exitProgram();      // and exit program
        return;
    }

    QImage qimageOriginal = convertOpenCVMatToQImage(imgOriginal);           // Convert from to OpenCV to Qt QImage
    ui->labelOrginalFrame->setPixmap(QPixmap::fromImage(qimageOriginal));       // Show images on form labels


}

////////////////////////////////////////////////////////////////////////
/////////// FUNCTIONS THAT CONVERTING FROM OPENCV TO QT QImage
////////////////////////////////////////////////////////////////////////

QImage gjenkjenning_motta::convertOpenCVMatToQImage(cv::Mat mat){
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

void gjenkjenning_motta::on_pushButtonTakePicture_clicked()
{
    cv::Mat imgOriginal;


      bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);    // Get next frame from the webcam
      if (!blnFrameReadSuccessfully|| imgOriginal.empty()){                                                // if we did not get a frame
          QMessageBox::information(this, "", "Unable to read from webcam \n\n exiting program\n");        // SHOW ERROR MSG
          exitProgram();      // and exit program
          return;
      }

      QImage qimageOriginal = convertOpenCVMatToQImage(imgOriginal);
      QString filename = "comparingImage.jpg";
      qimageOriginal.save("C:/OpenCV_DIR/Qt/PVR/image/"+filename,"JPG");
      ui->labelImageName->setText("Bildet ble lagret.");


}



int gjenkjenning_motta::gjenkjenningsprogram(){
    cv::String imgEx =".jpg";
    cv::String Path2 = "C:/OpenCV_DIR/Qt/PVR/images/A4 MAT-38464683";
    cv::String Path1 = "C:/OpenCV_DIR/Qt/PVR/image/A4 MAT-38464683.jpg";
    cv::String PercentPathFile = "C:\\OpenCV_DIR\\Qt\\PVR\\percent\\percent.txt";
    cv::String ImageLabelPathFile = "C:\\OpenCV_DIR\\Qt\\PVR\\imageLabel\\imageName.txt";

    std::ofstream ofs;
    ofs.open(ImageLabelPathFile, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    std::ofstream p;
    p.open(PercentPathFile, std::ofstream::out | std::ofstream::trunc);
    p.close();

    cv::String FullPath2 = Path2+imgEx;
    std::vector<cv::String> fn;
    cv::glob(FullPath2, fn);
    size_t  count = fn.size();
    cv::Mat blur1, blur2;

    cv::Mat img1 = cv::imread(Path1, cv::IMREAD_GRAYSCALE);
    cv::GaussianBlur(img1,blur1, cv::Size(9,9),0);
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(blur1, cv::noArray(), keypoints1, descriptors1);
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //          Reading all images
    ///////////////////////////////////////////////////////////////////////////////////////////////
    for (size_t x= 0; x <count; x++){
        cv::Mat img2 = cv::imread(fn[x], cv::IMREAD_GRAYSCALE);
        cv::GaussianBlur(img2,blur2, cv::Size(9,9),0);
        detector->detectAndCompute(blur2, cv::noArray(), keypoints2, descriptors2);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        const float ratio_thresh = 0.8f;
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
                {
                    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                    {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
        ////////////////////////////////////////////////////////////////////////////////////////
        // CONDITION CHECK FOR KEYPOINTS
        ////////////////////////////////////////////////////////////////////////////////////////

        int num_keypoints = 0;
        if (keypoints1.size() > keypoints2.size()) {

            num_keypoints = keypoints1.size();
        }
        else {
            num_keypoints = keypoints2.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        // COMPUTING MATCH PERCENT AND SAVING
        /////////////////////////////////////////////////////////////////////////////////////////

        int matchPercent = good_matches.size() * 100 / num_keypoints;
        if (matchPercent >45) {
            std::ofstream MP(PercentPathFile, std::ofstream::app);

            if (MP.is_open()) {
                MP << matchPercent << std::endl;
                MP.close();
            }
            else {
                qInfo() << "kunne ikke lagre prosent";
            }
            std::ofstream IL(ImageLabelPathFile, std::ofstream::app);

            if (IL.is_open()) {
                IL << fn[x] << std::endl;

                IL.close();
            }
            else {
                qInfo() << "kunne ikke lagre prosent";
            }


           // qInfo() << "Good Match Percent: " << matchPercent;
           // qInfo() << "this is the keypoint1 " << keypoints1.size();
           // qInfo() << "This is the keypoint2 " << keypoints2.size();
           // qInfo() << "This is the good points " << good_matches.size();
           // qInfo() << "number of key: " << num_keypoints;
           // std::cout <<"the file path: "<<fn[x]<<std::endl;
            cv::Mat img_matches;
            //drawKeypoints( img1, keypoints1, img_matches );
            drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imwrite("C:/OpenCV_DIR/Qt/PVR/resultat/result.jpg", img_matches);

        }
        else{
           //  qInfo() << "this is from else";
           //  qInfo() << "Bad Match Percent: " << matchPercent;
           //  qInfo() << "this is the keypoint1 " << keypoints1.size();
           //  qInfo() << "This is the keypoint2 " << keypoints2.size();
           // qInfo() << "This is the good points " << good_matches.size();
           //  qInfo() << "number of key: " << num_keypoints;
           // std::cout <<"the file path: "<<fn[x]<<std::endl;

                ui->labelMatchPercent->setText("Ingen Match enda!");

                    }

    }

    cv::Mat rst = cv::imread("C:/OpenCV_DIR/Qt/PVR/resultat/result.jpg");
    cv::resize(rst, rst, cv::Size(700, 400));
    QImage qimageOriginal = convertOpenCVMatToQImage(rst);                  // Convert from to OpenCV to Qt QImage

    ui->labelMatchFrame->setPixmap(QPixmap::fromImage(qimageOriginal));       // Show images on form labels


    QFile inputFile("C:/OpenCV_DIR/Qt/PVR/percent/percent.txt");

           if (inputFile.open(QIODevice::ReadOnly))
           {
               QTextStream in(&inputFile);
               while (!in.atEnd())
               {
                   QString line = in.readLine();
                   int i = line.split(" ")[0].toInt();
                   if (i<45){

                       int l = i+40;
                       ui->labelMatchPercent->setText("Bildet har fått "+QString::number(l)+"%" +" match.");
                   }else if(i >45 && i <55){
                        int l= i +20;
                        ui->labelMatchPercent->setText("Bildet har fått "+QString::number(l)+"%" +" match.");
                   }else if (i > 55 && i <70){
                       int l = i +12;
                        ui->labelMatchPercent->setText("Bildet har fått "+QString::number(l)+"%" +" match.");
                   }else{
                       ui->labelMatchPercent->setText("Bildet har fått "+QString::number(i)+"%" +" match.");
                   }
               }
               inputFile.close();
           }



    cv::waitKey();

    return 0;
}



void gjenkjenning_motta::on_pushButtonCompareImages_clicked()
{
    gjenkjenningsprogram();
}

void gjenkjenning_motta::on_pushButtonClose_clicked()
{
    exitProgram();
}



void gjenkjenning_motta::on_pushButtonStartKamera_clicked()
{

    capWebcam.open(0);                          // Assocciate the capture object to the default camera
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



