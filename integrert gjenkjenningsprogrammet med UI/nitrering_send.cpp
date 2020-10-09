#include "nitrering_send.h"
#include "ui_nitrering_send.h"

nitrering_send::nitrering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::nitrering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

nitrering_send::~nitrering_send()
{
    delete ui;
}
void nitrering_send::on_thune_produkter_raufoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void nitrering_send::on_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void nitrering_send::on_tilbake_0_nitrering_bedrifter_clicked()
{
    emit tilbake_0_nitrering_bedrifter_clicked();
}

void nitrering_send::on_tilbake_1_nitrering_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void nitrering_send::on_tilbake_2_nitrering_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
