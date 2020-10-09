#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowState(Qt::WindowMaximized);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - hjem - Logg inn
    this->setWindowTitle("Parts Vision Recognition");

    QPixmap pix("C:/Users/Kirisan/Pictures/PVR_local/telogo2.png");  //310x78
    ui->logo_tronrud->setPixmap(pix);
    QPixmap pix2("C:/Users/Kirisan/Pictures/PVR_local/pvr.png");  //178x100
    ui->logo_pvr->setPixmap(pix2);
    QPixmap pix4("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr->setPixmap(pix4);

    //Home ikon i venstre hjørne
    QPixmap pix3("C:/Users/Kirisan/Pictures/PVR_local/x_home.png");
    QIcon ButtonIcon(pix3);
    ui->x_1_main->setIcon(ButtonIcon);
    ui->x_1_main->setIconSize(pix3.rect().size());
    ui->x_1_main->setFixedSize(pix3.rect().size());


    ui->stackedWidget->insertWidget(2, &_Motta);    //setter inn MOTTAR.cpp i index 2, side 3
    ui->stackedWidget->insertWidget(3, &_Send);    //Setter inn SENDER.cpp i index 3, side 4


    connect(&_Send, SIGNAL(x_0_send_clicked()), this, SLOT(x_0_send_hjem()));
    connect(&_Send, SIGNAL(tilbake_0_send_clicked()), this, SLOT(tilbake_0_send()));

    connect(&_Motta, SIGNAL(x_0_motta_clicked()), this, SLOT(x_0_motta_hjem()));
    connect(&_Motta, SIGNAL(tilbake_0_motta_clicked()), this, SLOT(tilbake_0_motta()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

// Hjem: Logg inn siden: neste side
void MainWindow::on_neste_1_clicked()
{
    ui->stackedWidget->setCurrentIndex(1); //index 1 = page 2
}

// Side 2 -> side 1 i behandling.cpp: Ved valg: SEND
void MainWindow::on_send_clicked()
{
    ui->stackedWidget->setCurrentIndex(3); // index 2 = page 1 i sender.cpp
}


void MainWindow::on_motta_clicked()
{
    ui->stackedWidget->setCurrentIndex(2);
}


// Tillegs widgets: --------------------------------------------------------------------------------------------------------
// sender.cpp widgets:------------------------------------------------------------------------------------------------------

// X for avslutt: side 2 (SEND og MOTTA)
void MainWindow::on_x_1_main_clicked()
{
    ui->stackedWidget->setCurrentIndex(0); // Tilbake til logg inn siden.
}

// x for avslutt: side 1 i sender.cpp. (SEND)
void MainWindow::x_0_send_hjem()
{
    ui->stackedWidget->setCurrentIndex(0); // tilbake til logg inn siden fra sender.cpp
}

// tilbake til index 0 i sender.cpp.: (SEND og MOTTA)
void MainWindow::tilbake_0_send()
{
    ui->stackedWidget->setCurrentIndex(1);
}

// mottar.cpp widgets:------------------------------------------------------------------------------------------------------

// x for avslutt: side 1 i mottar.cpp: (Motta)
void MainWindow::x_0_motta_hjem()
{
    ui->stackedWidget->setCurrentIndex(0); // tilbake til logg inn siden fra mottar.cpp
}

void MainWindow::tilbake_0_motta()
{
  ui->stackedWidget->setCurrentIndex(1);
}






