#ifndef GJENKJENNING_SEND_H
#define GJENKJENNING_SEND_H

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
class gjenkjenning_send;
}

class gjenkjenning_send : public QWidget
{
    Q_OBJECT

public slots:
       void  processFrameAndUpdateGUI();        // Function prototype

public:
    explicit gjenkjenning_send(QWidget *parent = nullptr);
    ~gjenkjenning_send();

private slots:
    void on_pushButtonScanBarcode_clicked();

    void on_pushButtonTakePicture_clicked();

    void on_pushButtonClose_clicked();

    void on_pushButtonStartKamera_clicked();

private:
    Ui::gjenkjenning_send *ui;


    cv::VideoCapture capWebcam;                 // Capture object to use with webcam

    QTimer *qtimer;                             // timer for processFrameAndUpdateGUI()

    QImage gjenkjenning_send::convertOpenCVMatToQImage(cv::Mat mat);            // Function prototype

    void gjenkjenning_send::exitProgram();

    int gjenkjenning_send::decodeBarcode();
    int gjenkjenning_send::encodeBarcode();

};

#endif // GJENKJENNING_SEND_H
