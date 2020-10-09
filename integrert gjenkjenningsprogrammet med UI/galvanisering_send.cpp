#include "galvanisering_send.h"
#include "ui_galvanisering_send.h"

galvanisering_send::galvanisering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::galvanisering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);
    ui->logo_tepvr_3->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

galvanisering_send::~galvanisering_send()
{
    delete ui;
}

void galvanisering_send::on_br_berntsen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void galvanisering_send::on_gundersen_galvano_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void galvanisering_send::on_duozink_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void galvanisering_send::on_tilbake_0_galvanisering_bedrifter_clicked()
{
    emit tilbake_0_galvanisering_bedrifter_clicked();
}

void galvanisering_send::on_tilbake_1_galvanisering_br_berntsen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void galvanisering_send::on_tilbake_2_galvanisering_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void galvanisering_send::on_tilbake_3_galvanisering_duozink_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


