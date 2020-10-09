#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStackedWidget>
#include <sender.h>
#include <mottar.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_neste_1_clicked(); //index 0, side 1 - fra logg inn til SEND og MOTTA - KNAPP (neste)

    void on_x_1_main_clicked();

    void on_send_clicked();
    void on_motta_clicked();


    //tilleggs funksjoner fra sub filer-------------------------------------------------------------------------------------------------------
    void x_0_send_hjem();                //index 0, side 1 - sender.cpp -
    void tilbake_0_send();               //index 0, side 1 - sender.cpp -

    void x_0_motta_hjem();
    void tilbake_0_motta();


private:
    Ui::MainWindow *ui;
    class sender _Send;   //objekt
    class mottar _Motta;  //objekt
};
#endif // MAINWINDOW_H
