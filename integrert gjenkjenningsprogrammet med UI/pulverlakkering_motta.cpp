#include "pulverlakkering_motta.h"
#include "ui_pulverlakkering_motta.h"

pulverlakkering_motta::pulverlakkering_motta(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::pulverlakkering_motta)
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

pulverlakkering_motta::~pulverlakkering_motta()
{
    delete ui;
}

void pulverlakkering_motta::on_ac_teknikk_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void pulverlakkering_motta::on_moss_jern_og_stanse_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void pulverlakkering_motta::on_rustfrie_berg_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void pulverlakkering_motta::on_ivar_braathen_mekaniske_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void pulverlakkering_motta::on_tilbake_0_pulver_bedrifter_clicked()
{
    emit tilbake_0_pulver_bedrifter_clicked();
}

void pulverlakkering_motta::on_tilbake_1_pulver_ac_teknikk_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void pulverlakkering_motta::on_tilbake_2_pulver_moss_jern_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void pulverlakkering_motta::on_tilbake_3_pulver_rustfrieBerg_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void pulverlakkering_motta::on_tilbake_4_pulver_ivarBraathen_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}
