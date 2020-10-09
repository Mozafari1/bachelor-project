#include "glass_blaasing_motta.h"
#include "ui_glass_blaasing_motta.h"

glass_blaasing_motta::glass_blaasing_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::glass_blaasing_motta)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);

    //Tronrud logo i knappen
    QPixmap pix("C:/Users/Kirisan/Pictures/PVR_local/te_glass.png");
    QIcon ButtonIcon(pix);
    ui->tronrud->setIcon(ButtonIcon);
    ui->tronrud->setIconSize(pix.rect().size());
    ui->tronrud->setFixedSize(pix.rect().size());

    ui->stackedWidget->insertWidget(0, &_GjenkjennMotta);
}

glass_blaasing_motta::~glass_blaasing_motta()
{
    delete ui;
}

void glass_blaasing_motta::on_tronrud_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void glass_blaasing_motta::on_tilbake_0_glass_blaasing_bedrifter_clicked()
{
    emit tilbake_0_glass_blaasing_bedrifter_clicked();
}
void glass_blaasing_motta::on_tilbake_1_glass_blaasing_tronrud_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
