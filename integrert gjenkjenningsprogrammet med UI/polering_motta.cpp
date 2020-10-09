#include "polering_motta.h"
#include "ui_polering_motta.h"

polering_motta::polering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::polering_motta)
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

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

polering_motta::~polering_motta()
{
    delete ui;
}

void polering_motta::on_tronrud_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void polering_motta::on_honefoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void polering_motta::on_tilbake_0_polering_bedrifter_clicked()
{
    emit tilbake_0_polering_bedrifter_clicked();
}

void polering_motta::on_tilbake_1_polering_tronrud_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void polering_motta::on_tilbake_2_polering_honefoss_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


