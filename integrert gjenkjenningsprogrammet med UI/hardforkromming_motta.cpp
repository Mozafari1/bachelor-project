#include "hardforkromming_motta.h"
#include "ui_hardforkromming_motta.h"

hardforkromming_motta::hardforkromming_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::hardforkromming_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

hardforkromming_motta::~hardforkromming_motta()
{
    delete ui;
}
void hardforkromming_motta::on_norsk_everit_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void hardforkromming_motta::on_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void hardforkromming_motta::on_tilbake_0_harforkromming_bedrifter_clicked()
{
    emit tilbake_0_hardforkromming_bedrifter_clicked();
}

void hardforkromming_motta::on_tilbake_1_hardforkromming_norsk_everit_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void hardforkromming_motta::on_tilbake_2_hardforkromming_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


