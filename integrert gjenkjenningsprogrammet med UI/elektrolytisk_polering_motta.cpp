#include "elektrolytisk_polering_motta.h"
#include "ui_elektrolytisk_polering_motta.h"

elektrolytisk_polering_motta::elektrolytisk_polering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::elektrolytisk_polering_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

elektrolytisk_polering_motta::~elektrolytisk_polering_motta()
{
    delete ui;
}

void elektrolytisk_polering_motta::on_gundersen_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void elektrolytisk_polering_motta::on_tilbake_0_elektrolytisk_polering__bedrifter_clicked()
{
    emit tilbake_0_elektrolytisk_polering_bedrifter_clicked();
}

void elektrolytisk_polering_motta::on_tilbake_1_elektrolytisk_polering_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}



