#include "kromatering_motta.h"
#include "ui_kromatering_motta.h"

kromatering_motta::kromatering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::kromatering_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);
    ui->logo_tepvr_3->setPixmap(pix0);
    ui->logo_tepvr_4->setPixmap(pix0);
    ui->logo_tepvr_5->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

kromatering_motta::~kromatering_motta()
{
    delete ui;
}

void kromatering_motta::on_eloksal_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void kromatering_motta::on_gundersen_galvano_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void kromatering_motta::on_gundersen_galvano_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void kromatering_motta::on_eiker_overflateteknikk_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void kromatering_motta::on_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void kromatering_motta::on_tilbake_0_kromatering_bedrifter_clicked()
{
    emit tilbake_0_kromatering_bedrifter_clicked();
}

void kromatering_motta::on_tilbake_1_eloksal_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void kromatering_motta::on_tilbake_2_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void kromatering_motta::on_tilbake_3_gundersen_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void kromatering_motta::on_tilbake_4_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void kromatering_motta::on_tilbake_5_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
