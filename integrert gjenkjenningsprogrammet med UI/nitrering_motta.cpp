#include "nitrering_motta.h"
#include "ui_nitrering_motta.h"

nitrering_motta::nitrering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::nitrering_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

nitrering_motta::~nitrering_motta()
{
    delete ui;
}

void nitrering_motta::on_thune_produkter_raufoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void nitrering_motta::on_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void nitrering_motta::on_tilbake_0_nitrering_bedrifter_clicked()
{
    emit tilbake_0_nitrering_bedrifter_clicked();
}

void nitrering_motta::on_tilbake_1_nitrering_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void nitrering_motta::on_tilbake_2_nitrering_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

