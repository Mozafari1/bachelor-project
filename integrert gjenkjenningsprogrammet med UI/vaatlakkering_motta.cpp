#include "vaatlakkering_motta.h"
#include "ui_vaatlakkering_motta.h"

vaatlakkering_motta::vaatlakkering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::vaatlakkering_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

vaatlakkering_motta::~vaatlakkering_motta()
{
    delete ui;
}

void vaatlakkering_motta::on_bilskadesenteret_ringerike_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void vaatlakkering_motta::on_lyseng_billakkering_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------

void vaatlakkering_motta::on_tilbake_0_vaat_bedrifter_clicked()
{
    emit tilbake_0_vaat_bedrifter_clicked();
}

void vaatlakkering_motta::on_tilbake_1_vaat_bilskade_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void vaatlakkering_motta::on_tilbake_2_vaat_lyseng_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
