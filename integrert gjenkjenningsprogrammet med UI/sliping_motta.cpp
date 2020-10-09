#include "sliping_motta.h"
#include "ui_sliping_motta.h"

sliping_motta::sliping_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::sliping_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

sliping_motta::~sliping_motta()
{
    delete ui;
}

void sliping_motta::on_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void sliping_motta::on_tilbake_0_sliping_bedrifter_clicked()
{
    emit tilbake_0_sliping_bedrifter_clicked();
}

void sliping_motta::on_tilbake_1_sliping_thune_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
