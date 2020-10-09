#include "syrevask_send.h"
#include "ui_syrevask_send.h"

syrevask_send::syrevask_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::syrevask_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);
    ui->logo_tepvr_3->setPixmap(pix0);
    ui->logo_tepvr_4->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

syrevask_send::~syrevask_send()
{
    delete ui;
}

void syrevask_send::on_tl_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_send::on_gundersen_galvano_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_send::on_eiker_overflateteknikk_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_send::on_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void syrevask_send::on_tilbake_0_syrevask_bedrifter_clicked()
{
    emit tilbake_0_syrevask_bedrifter_clicked();
}
void syrevask_send::on_tilbake_1_syrevask_tl_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_send::on_tilbake_2_syrevask_gundersen_vestby_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_send::on_tilbake_3_syrevask_eiker_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void syrevask_send::on_tilbake_4_syrevask_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}











