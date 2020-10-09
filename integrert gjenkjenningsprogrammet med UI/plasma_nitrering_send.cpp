#include "plasma_nitrering_send.h"
#include "ui_plasma_nitrering_send.h"

plasma_nitrering_send::plasma_nitrering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::plasma_nitrering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

plasma_nitrering_send::~plasma_nitrering_send()
{
    delete ui;
}

void plasma_nitrering_send::on_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void plasma_nitrering_send::on_tilbake_0_plasma_nitrering_bedrifter_clicked()
{
    emit tilbake_0_plasma_nitrering_bedrifter_clicked();
}

void plasma_nitrering_send::on_tilbake_1_plasma_nitrering_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


