#include "brunering_motta.h"
#include "ui_brunering_motta.h"

brunering_motta::brunering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::brunering_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

brunering_motta::~brunering_motta()
{
    delete ui;
}

void brunering_motta::on_borsemakern_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void brunering_motta::on_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void brunering_motta::on_tilbake_0_brunering_bedrifter_clicked()
{
    emit tilbake_0_brunering_bedrifter_clicked();
}

void brunering_motta::on_tilbake_1_brunering_borsemakern_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void brunering_motta::on_tilbake_2_brunering_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}



