#include "elzink_motta.h"
#include "ui_elzink_motta.h"

elzink_motta::elzink_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::elzink_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);
    ui->logo_tepvr_3->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

elzink_motta::~elzink_motta()
{
    delete ui;
}

void elzink_motta::on_gundersen_galvano_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void elzink_motta::on_eiker_overflateteknikk_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void elzink_motta::on_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void elzink_motta::on_tilbake_0_elzink_bedrifter_clicked()
{
    emit tilbake_0_elzink_bedrifter_clicked();
}

void elzink_motta::on_tilbake_1_elzink_gundersen_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void elzink_motta::on_tilbake_2_elzink_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void elzink_motta::on_tilbake_3_elzink_honefoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
