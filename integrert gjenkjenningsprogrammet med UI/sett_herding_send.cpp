#include "sett_herding_send.h"
#include "ui_sett_herding_send.h"

sett_herding_send::sett_herding_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::sett_herding_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

sett_herding_send::~sett_herding_send()
{
    delete ui;
}

void sett_herding_send::on_thune_produkter_raufoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void sett_herding_send::on_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void sett_herding_send::on_tilbake_0_sett_herding_bedrifter_clicked()
{
    emit tilbake_0_sett_herding_bedrifter_clicked();
}

void sett_herding_send::on_tilbake_1_sett_herding_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void sett_herding_send::on_tilbake_2_sett_herding_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
