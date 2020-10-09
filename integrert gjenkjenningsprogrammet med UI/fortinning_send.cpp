#include "fortinning_send.h"
#include "ui_fortinning_send.h"

fortinning_send::fortinning_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::fortinning_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

fortinning_send::~fortinning_send()
{
    delete ui;
}

void fortinning_send::on_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void fortinning_send::on_tilbake_0_fortinning_bedrifter_clicked()
{
    emit tilbake_0_fortinning_bedrifter_clicked();
}

void fortinning_send::on_tilbake_1_fortinning_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


