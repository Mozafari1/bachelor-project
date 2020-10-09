#include "polering_send.h"
#include "ui_polering_send.h"

polering_send::polering_send(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::polering_send)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);

    //Tronrud logo i knappen
    QPixmap pix("C:/Users/Kirisan/Pictures/PVR_local/te_glass.png");
    QIcon ButtonIcon(pix);
    ui->tronrud->setIcon(ButtonIcon);
    ui->tronrud->setIconSize(pix.rect().size());
    ui->tronrud->setFixedSize(pix.rect().size());

    ui->stackedWidget->insertWidget(0, &_GjenkjennSend);
}

polering_send::~polering_send()
{
    delete ui;
}

void polering_send::on_tronrud_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void polering_send::on_honefoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void polering_send::on_tilbake_0_polering_bedrifter_clicked()
{
    emit tilbake_0_polering_bedrifter_clicked();
}

void polering_send::on_tilbake_1_polering_tronrud_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void polering_send::on_tilbake_2_polering_honefoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


