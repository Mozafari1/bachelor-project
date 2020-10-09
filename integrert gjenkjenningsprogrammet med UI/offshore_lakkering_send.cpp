#include "offshore_lakkering_send.h"
#include "ui_offshore_lakkering_send.h"

offshore_lakkering_send::offshore_lakkering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::offshore_lakkering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

offshore_lakkering_send::~offshore_lakkering_send()
{
    delete ui;
}

void offshore_lakkering_send::on_nct_clicked()  //Norwegian Coating Technologies (NCT)
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------


void offshore_lakkering_send::on_tilbake_0_offshore_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void offshore_lakkering_send::on_tilbake_0_offshore_bedrifter_clicked()
{
    emit tilbake_0_offshore_bedrifter_clicked();
}
