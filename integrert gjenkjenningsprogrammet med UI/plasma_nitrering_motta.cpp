#include "plasma_nitrering_motta.h"
#include "ui_plasma_nitrering_motta.h"

plasma_nitrering_motta::plasma_nitrering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::plasma_nitrering_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

plasma_nitrering_motta::~plasma_nitrering_motta()
{
    delete ui;
}

void plasma_nitrering_motta::on_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void plasma_nitrering_motta::on_tilbake_0_plasma_nitrering_bedrifter_clicked()
{
    emit tilbake_0_plasma_nitrering_bedrifter_clicked();
}

void plasma_nitrering_motta::on_tilbake_1_plasma_nitrering_bodycote_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


