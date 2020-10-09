#include "galvanisering_motta.h"
#include "ui_galvanisering_motta.h"

galvanisering_motta::galvanisering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::galvanisering_motta)
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

galvanisering_motta::~galvanisering_motta()
{
    delete ui;
}

void galvanisering_motta::on_br_berntsen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void galvanisering_motta::on_gundersen_galvano_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void galvanisering_motta::on_duozink_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void galvanisering_motta::on_tilbake_0_galvanisering_bedrifter_clicked()
{
    emit tilbake_0_galvanisering_bedrifter_clicked();
}

void galvanisering_motta::on_tilbake_1_galvanisering_br_berntsen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void galvanisering_motta::on_tilbake_2_galvanisering_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void galvanisering_motta::on_tilbake_3_galvanisering_duozink_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

