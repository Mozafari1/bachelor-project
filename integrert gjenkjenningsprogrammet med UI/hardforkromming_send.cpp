#include "hardforkromming_send.h"
#include "ui_hardforkromming_send.h"

hardforkromming_send::hardforkromming_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::hardforkromming_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

hardforkromming_send::~hardforkromming_send()
{
    delete ui;
}

void hardforkromming_send::on_norsk_everit_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void hardforkromming_send::on_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void hardforkromming_send::on_tilbake_0_harforkromming_bedrifter_clicked()
{
    emit tilbake_0_hardforkromming_bedrifter_clicked();
}

void hardforkromming_send::on_tilbake_1_hardforkromming_norsk_everit_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void hardforkromming_send::on_tilbake_2_hardforkromming_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


