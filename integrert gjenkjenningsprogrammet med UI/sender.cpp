#include "sender.h"
#include "ui_sender.h"

sender::sender(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::sender)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - SEND

    QPixmap pix0("C:/Users/Kirisan/Pictures/PVR_local/tePVRlogo.png"); //481x94
    ui->logo_tepvr_0->setPixmap(pix0);
    ui->logo_tepvr_1->setPixmap(pix0);
    ui->logo_tepvr_2->setPixmap(pix0);
    ui->logo_tepvr_3->setPixmap(pix0);
    ui->logo_tepvr_4->setPixmap(pix0);
    ui->logo_tepvr_5->setPixmap(pix0);

    //Home ikon i venstre hjørne, x_0_send
    QPixmap pix("C:/Users/Kirisan/Pictures/PVR_local/x_home.png");
    QIcon ButtonIcon(pix);
    ui->x_0_send->setIcon(ButtonIcon);
    ui->x_0_send->setIconSize(pix.rect().size());
    ui->x_0_send->setFixedSize(pix.rect().size());

    ui->stackedWidget->insertWidget(6, &_Vaatlakk);                     //setter inn vaatlakkering.cpp i index 6, side 7
    ui->stackedWidget->insertWidget(7, &_Pulverlakk);                   //setter inn pulverlakkering.cpp i index 7, side 8
    ui->stackedWidget->insertWidget(8, &_Offshorelakk);                 //setter inn offshore_lakkering.cpp i index 8, side 9
    ui->stackedWidget->insertWidget(9, &_Eloks);                        //Setter inn eloksering.cpp i index 9, side 10
    ui->stackedWidget->insertWidget(10, &_Kromater);                    //Setter inn kromatering.cpp i index 10, side 11
    ui->stackedWidget->insertWidget(11, &_Elzink);                      //Setter inn elzink.cpp i index 11, side 12
    ui->stackedWidget->insertWidget(12, &_Hardforkrom);                 //Setter inn hardforkromming.cpp i index 12, side 13
    ui->stackedWidget->insertWidget(13, &_ElektroFornikling);           //Setter inn elektrolytisk_fornikling.cpp i index 13, side 14
    ui->stackedWidget->insertWidget(14, &_Fosfat);                      //Setter inn fosfatering.cpp i index 14, side 15
    ui->stackedWidget->insertWidget(15, &_Galv);                        //Setter inn galvanisering.cpp i index 15, side 16
    ui->stackedWidget->insertWidget(16, &_Syre);                        //Setter inn syrevask.cpp i index 16, side 17
    ui->stackedWidget->insertWidget(17, &_Brun);                        //Setter inn brunering.cpp i index 17, side 18
    ui->stackedWidget->insertWidget(18, &_Herd);                        //Setter inn herding.cpp i index 18, side 19
    ui->stackedWidget->insertWidget(19, &_SettHerd);                    //Setter inn sett_herding.cpp i index 19, side 20
    ui->stackedWidget->insertWidget(20, &_Nitrer);                      //Setter inn nitrering.cpp i index 20, side 21
    ui->stackedWidget->insertWidget(21, &_PlasmaNitrer);                //Setter inn plasma_nitrering.cpp i index 21, side 22
    ui->stackedWidget->insertWidget(22, &_Sliping);                     //Setter inn sliping.cpp i index 22, side 23
    ui->stackedWidget->insertWidget(23, &_Traad);                       //Setter inn traadskjaering.cpp i index 23, side 24
    ui->stackedWidget->insertWidget(24, &_Glass);                       //Setter inn glass_blaasing.cpp i index 24, side 25
    ui->stackedWidget->insertWidget(25, &_Sand);                        //Setter inn sandblaasing.cpp i index 25, side 26
    ui->stackedWidget->insertWidget(26, &_Poler);                       //Setter inn polering.cpp i index 27, side 28
    ui->stackedWidget->insertWidget(27, &_ElektroPolering);             //Setter inn elektrolytisk_polering.cpp i index 28, side 29
    ui->stackedWidget->insertWidget(28, &_Forsolv);                     //Setter inn forsolving.cpp i index 29, side 30
    ui->stackedWidget->insertWidget(29, &_Gull);                        //Setter inn gullbelegging.cpp i index 30, side 31
    ui->stackedWidget->insertWidget(30, &_Fortinning);                  //Setter inn fortinning.cpp i index 31, side 32

    //Lakkering
    connect(&_Vaatlakk, SIGNAL(tilbake_0_vaat_bedrifter_clicked()), this, SLOT(tilbake_0_vaat_bedrifter()));
    connect(&_Pulverlakk, SIGNAL(tilbake_0_pulver_bedrifter_clicked()), this, SLOT(tilbake_0_pulver_bedrifter()));
    connect(&_Offshorelakk, SIGNAL(tilbake_0_offshore_bedrifter_clicked()), this, SLOT(tilbake_0_offshore_bedrifter()));

    //elektrokjemisk_metode
    connect(&_Eloks, SIGNAL(tilbake_0_eloksering_bedrifter_clicked()), this, SLOT(tilbake_0_eloksering_bedrifter()));
    connect(&_Kromater, SIGNAL(tilbake_0_kromatering_bedrifter_clicked()), this, SLOT(tilbake_0_kromatering_bedrifter()));
    connect(&_Elzink, SIGNAL(tilbake_0_elzink_bedrifter_clicked()), this, SLOT(tilbake_0_elzink_bedrifter()));
    connect(&_Hardforkrom, SIGNAL(tilbake_0_hardforkromming_bedrifter_clicked()), this, SLOT(tilbake_0_hardforkromming_bedrifter()));
    connect(&_ElektroFornikling, SIGNAL(tilbake_0_elektrolytisk_fornikling_bedrifter_clicked()), this, SLOT(tilbake_0_elektrolytisk_fornikling_bedrifter()));

    //kjemisk_metode
    connect(&_Fosfat, SIGNAL(tilbake_0_fosfatering_bedrifter_clicked()), this, SLOT(tilbake_0_fosfatering_bedrifter()));
    connect(&_Galv, SIGNAL(tilbake_0_galvanisering_bedrifter_clicked()), this, SLOT(tilbake_0_galvanisering_bedrifter()));
    connect(&_Syre, SIGNAL(tilbake_0_syrevask_bedrifter_clicked()), this, SLOT(tilbake_0_syrevask_bedrifter()));
    connect(&_Brun, SIGNAL(tilbake_0_brunering_bedrifter_clicked()), this, SLOT(tilbake_0_brunering_bedrifter()));

    //herding
    connect(&_Herd, SIGNAL(tilbake_0_herding_bedrifter_clicked()), this, SLOT(tilbake_0_herding_bedrifter()));
    connect(&_SettHerd, SIGNAL(tilbake_0_sett_herding_bedrifter_clicked()), this, SLOT(tilbake_0_sett_herding_bedrifter()));
    connect(&_Nitrer, SIGNAL(tilbake_0_nitrering_bedrifter_clicked()), this, SLOT(tilbake_0_nitrering_bedrifter()));
    connect(&_PlasmaNitrer, SIGNAL(tilbake_0_plasma_nitrering_bedrifter_clicked()), this, SLOT(tilbake_0_plasma_nitrering_bedrifter()));

    //annet
    connect(&_Sliping, SIGNAL(tilbake_0_sliping_bedrifter_clicked()), this, SLOT(tilbake_0_sliping_bedrifter()));
    connect(&_Traad, SIGNAL(tilbake_0_traadskjaering_bedrifter_clicked()), this, SLOT(tilbake_0_traadskjaering_bedrifter()));
    connect(&_Glass, SIGNAL(tilbake_0_glass_blaasing_bedrifter_clicked()), this, SLOT(tilbake_0_glass_blaasing_bedrifter()));
    connect(&_Sand, SIGNAL(tilbake_0_sandblaasing_bedrifter_clicked()), this, SLOT(tilbake_0_sandblaasing_bedrifter()));
    connect(&_Poler, SIGNAL(tilbake_0_polering_bedrifter_clicked()), this, SLOT(tilbake_0_polering_bedrifter()));
    connect(&_ElektroPolering, SIGNAL(tilbake_0_elektrolytisk_polering_bedrifter_clicked()), this, SLOT(tilbake_0_elektrolytisk_polering_bedrifter()));
    connect(&_Forsolv, SIGNAL(tilbake_0_forsolving_bedrifter_clicked()), this, SLOT(tilbake_0_forsolving_bedrifter()));
    connect(&_Gull, SIGNAL(tilbake_0_gullbelegging_bedrifter_clicked()), this, SLOT(tilbake_0_gullbelegging_bedrifter()));
    connect(&_Fortinning, SIGNAL(tilbake_0_fortinning_bedrifter_clicked()), this, SLOT(tilbake_0_fortinning_bedrifter()));
}

sender::~sender()
{
    delete ui;
}

//lakkeringsvindu-------------------------------------------------------------------------------
void sender::on_lakkering_clicked()
{
    ui->stackedWidget->setCurrentIndex(1); //index 1, side 2 - Lakkering
}
    void sender::on_vaatlakkering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(6);
    }

    void sender::tilbake_0_vaat_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(1);
    }

    void sender::on_pulverlakkering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(7);
    }

    void sender::tilbake_0_pulver_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(1);
    }

    void sender::on_offshore_lakkering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(8);
    }

    void sender::tilbake_0_offshore_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(1);
    }

//elektrokjemisk metode vindu-------------------------------------------------------------------------------
void sender::on_elektrokjemisk_metode_clicked()
{
    ui->stackedWidget->setCurrentIndex(2); //index 2, side 3 - Elektrokjemisk metode
}

    void sender::on_eloksering_alu_clicked()
    {
        ui->stackedWidget->setCurrentIndex(9);
    }
    void sender::tilbake_0_eloksering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(2); //index 2, side 3 i sender.cpp - Elektrokjemisk metode
    }

    void sender::on_kromatering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(10);
    }
    void sender::tilbake_0_kromatering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void sender::on_elzink_clicked()
    {
        ui->stackedWidget->setCurrentIndex(11);
    }
    void sender::tilbake_0_elzink_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void sender::on_hardforkromming_clicked()
    {
        ui->stackedWidget->setCurrentIndex(12);
    }
    void sender::tilbake_0_hardforkromming_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void sender::on_elektrolytisk_fornikling_clicked()
    {
        ui->stackedWidget->setCurrentIndex(13);
    }
    void sender::tilbake_0_elektrolytisk_fornikling_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

//kjemisk metode vindu-------------------------------------------------------------------------------
void sender::on_kjemisk_metode_clicked()
{
    ui->stackedWidget->setCurrentIndex(3); //index 3, side 4 - Kjemisk metode
}
    void sender::on_fosfatering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(14);
    }
    void sender::tilbake_0_fosfatering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

    void sender::on_galvanisering_varmforsinking_clicked()
    {
        ui->stackedWidget->setCurrentIndex(15);
    }
    void sender::tilbake_0_galvanisering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

    void sender::on_syrevask_clicked()
    {
        ui->stackedWidget->setCurrentIndex(16);
    }
    void sender::tilbake_0_syrevask_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

    void sender::on_brunering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(17);
    }
    void sender::tilbake_0_brunering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

//herding vindu-------------------------------------------------------------------------------
void sender::on_herding_clicked()
{
    ui->stackedWidget->setCurrentIndex(4); //index 4, side 5 - Herding
}
    void sender::on_herd_herding_clicked()
    {
        ui->stackedWidget->setCurrentIndex(18);
    }
    void sender::tilbake_0_herding_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

    void sender::on_sett_herding_clicked()
    {
        ui->stackedWidget->setCurrentIndex(19);
    }
    void sender::tilbake_0_sett_herding_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

    void sender::on_nitrering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(20);
    }
    void sender::tilbake_0_nitrering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

    void sender::on_plasma_nitrering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(21);
    }
    void sender::tilbake_0_plasma_nitrering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

//annet vindu-------------------------------------------------------------------------------
void sender::on_annet_clicked()
{
    ui->stackedWidget->setCurrentIndex(5); //index5, side 6 - Annet
}
    void sender::on_sliping_clicked()
    {
        ui->stackedWidget->setCurrentIndex(22);
    }
    void sender::tilbake_0_sliping_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_traadskjaering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(23);
    }
    void sender::tilbake_0_traadskjaering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_glass_blaasing_clicked()
    {
        ui->stackedWidget->setCurrentIndex(24);
    }
    void sender::tilbake_0_glass_blaasing_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_sandblaasing_clicked()
    {
        ui->stackedWidget->setCurrentIndex(25);
    }
    void sender::tilbake_0_sandblaasing_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_polering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(26);
    }
    void sender::tilbake_0_polering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_elektrolytisk_polering_clicked()
    {
        ui->stackedWidget->setCurrentIndex(27);
    }
    void sender::tilbake_0_elektrolytisk_polering_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_forsolving_gullbelegging_clicked()
    {
        ui->stackedWidget->setCurrentIndex(28);
    }
    void sender::tilbake_0_forsolving_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_gullbelegging_clicked()
    {
        ui->stackedWidget->setCurrentIndex(29);
    }
    void sender::tilbake_0_gullbelegging_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void sender::on_fortinning_clicked()
    {
        ui->stackedWidget->setCurrentIndex(30);
    }
    void sender::tilbake_0_fortinning_bedrifter()
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void sender::on_x_0_send_clicked()                                     //Avslutt --> Hjem/Logg inn - SIGNAL
{
    emit x_0_send_clicked();                // avslutter og tilbake til hjem (index 0, side 1 i main).
}

void sender::on_tilbake_0_send_clicked()                               //tilbakeknapp i motta.cpp --> Hjem/logg inn - SIGNAL
{
    emit tilbake_0_send_clicked();          //ett hakk tilbake (index 0, side 1 i sender) - Destinasjon: SEND og MOTTA
}

void sender::on_tilbake_1_lakk_send_clicked()                          //tilbakeknapp i lakkering vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void sender::on_tilbake_2_elkj_send_clicked()                          //tilbakeknapp i elektrokjemisk_metode vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void sender::on_tilbake_3_kj_send_clicked()                            //tilbakeknapp i kjemisk_metode vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void sender::on_tilbake_4_herd_send_clicked()                          //tilbakeknapp i herding vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void sender::on_tilbake_5_annet_send_clicked()                         //tilbakeknapp i annet vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}
