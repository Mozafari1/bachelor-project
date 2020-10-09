#include "forsolving_send.h"
#include "ui_forsolving_send.h"

forsolving_send::forsolving_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::forsolving_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

forsolving_send::~forsolving_send()
{
    delete ui;
}

void forsolving_send::on_lie_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void forsolving_send::on_tilbake_0_forsolving__bedrifter_clicked()
{
    emit tilbake_0_forsolving_bedrifter_clicked();
}

void forsolving_send::on_tilbake_1_forsolving_lie_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


