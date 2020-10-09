#include "syrevask_motta.h"
#include "ui_syrevask_motta.h"

syrevask_motta::syrevask_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::syrevask_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);
    ui->logo_tepvr_3->setPixmap(pix0);
    ui->logo_tepvr_4->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

syrevask_motta::~syrevask_motta()
{
    delete ui;
}

void syrevask_motta::on_tl_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_motta::on_gundersen_galvano_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_motta::on_eiker_overflateteknikk_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_motta::on_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void syrevask_motta::on_tilbake_0_syrevask_bedrifter_clicked()
{
    emit tilbake_0_syrevask_bedrifter_clicked();
}
void syrevask_motta::on_tilbake_1_syrevask_tl_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_motta::on_tilbake_2_syrevask_gundersen_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_motta::on_tilbake_3_syrevask_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_motta::on_tilbake_4_syrevask_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}



