#include "gullbelegging_motta.h"
#include "ui_gullbelegging_motta.h"

gullbelegging_motta::gullbelegging_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::gullbelegging_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

gullbelegging_motta::~gullbelegging_motta()
{
    delete ui;
}

void gullbelegging_motta::on_jens_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void gullbelegging_motta::on_tilbake_0_gullbelegging__bedrifter_clicked()
{
    emit tilbake_0_gullbelegging_bedrifter_clicked();
}

void gullbelegging_motta::on_tilbake_1_gullbelegging_jens_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


