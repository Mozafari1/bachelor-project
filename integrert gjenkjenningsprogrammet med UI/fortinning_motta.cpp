#include "fortinning_motta.h"
#include "ui_fortinning_motta.h"

fortinning_motta::fortinning_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::fortinning_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

fortinning_motta::~fortinning_motta()
{
    delete ui;
}

void fortinning_motta::on_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void fortinning_motta::on_tilbake_0_fortinning_bedrifter_clicked()
{
    emit tilbake_0_fortinning_bedrifter_clicked();
}

void fortinning_motta::on_tilbake_1_fortinning_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


