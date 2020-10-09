#include "sliping_send.h"
#include "ui_sliping_send.h"

sliping_send::sliping_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::sliping_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

sliping_send::~sliping_send()
{
    delete ui;
}

void sliping_send::on_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void sliping_send::on_tilbake_0_sliping_bedrifter_clicked()
{
    emit tilbake_0_sliping_bedrifter_clicked();
}

void sliping_send::on_tilbake_1_sliping_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


