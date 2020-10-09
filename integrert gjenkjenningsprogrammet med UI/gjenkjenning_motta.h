#ifndef GJENKJENNING_MOTTA_H
#define GJENKJENNING_MOTTA_H

#include <QWidget>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<QtSql>
#include<QSqlDatabase>
////////////////////////////////////////////////////////////////////////////////////////////

namespace Ui {
class gjenkjenning_motta;
}

class gjenkjenning_motta : public QWidget
{
    Q_OBJECT
public slots:
       void  processFrameAndUpdateGUI();        // Function prototype

public:
    explicit gjenkjenning_motta(QWidget *parent = nullptr);
    ~gjenkjenning_motta();

private slots:


    void on_pushButtonTakePicture_clicked();

    void on_pushButtonCompareImages_clicked();

    void on_pushButtonClose_clicked();



    void on_pushButtonStartKamera_clicked();



private:
    Ui::gjenkjenning_motta *ui;


    cv::VideoCapture capWebcam;                 // Capture object to use with webcam

    QTimer *qtimer;                             // timer for processFrameAndUpdateGUI()

    QImage gjenkjenning_motta::convertOpenCVMatToQImage(cv::Mat mat);            // Function prototype

    void gjenkjenning_motta::exitProgram();
    int gjenkjenning_motta::gjenkjenningsprogram();

};

#endif // GJENKJENNING_MOTTA_H
