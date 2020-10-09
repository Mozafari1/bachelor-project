#include "brunering_send.h"
#include "ui_brunering_send.h"

brunering_send::brunering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::brunering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);

}

brunering_send::~brunering_send()
{
    delete ui;
}

void brunering_send::on_borsemakern_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void brunering_send::on_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void brunering_send::on_tilbake_0_brunering_bedrifter_clicked()
{
    emit tilbake_0_brunering_bedrifter_clicked();
}

void brunering_send::on_tilbake_1_brunering_borsemakern_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void brunering_send::on_tilbake_2_brunering_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


