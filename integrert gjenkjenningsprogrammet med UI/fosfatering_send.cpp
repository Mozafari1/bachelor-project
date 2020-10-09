#include "fosfatering_send.h"
#include "ui_fosfatering_send.h"

fosfatering_send::fosfatering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::fosfatering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

fosfatering_send::~fosfatering_send()
{
    delete ui;
}

void fosfatering_send::on_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
void fosfatering_send::on_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void fosfatering_send::on_tilbake_0_fosfatering_bedrifter_clicked()
{
    emit tilbake_0_fosfatering_bedrifter_clicked();
}

void fosfatering_send::on_tilbake_1_fosfatering_nct_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void fosfatering_send::on_tilbake_2_fosfatering_gundersen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


