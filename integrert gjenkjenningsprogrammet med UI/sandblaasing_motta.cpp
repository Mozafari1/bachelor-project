#include "sandblaasing_motta.h"
#include "ui_sandblaasing_motta.h"

sandblaasing_motta::sandblaasing_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::sandblaasing_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

sandblaasing_motta::~sandblaasing_motta()
{
    delete ui;
}

void sandblaasing_motta::on_bilskadesenteret_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void sandblaasing_motta::on_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void sandblaasing_motta::on_tilbake_0_sandblaasing_bedrifter_clicked()
{
    emit tilbake_0_sandblaasing_bedrifter_clicked();
}

void sandblaasing_motta::on_tilbake_1_sandblaasing_bilskadesenteret_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void sandblaasing_motta::on_tilbake_2_sandblaasing_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
