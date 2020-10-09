#include "mottar.h"
#include "ui_mottar.h"

mottar::mottar(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::mottar)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentIndex(0);  // start alltid på side 1 - MOTTA

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
    ui->x_0_motta->setIcon(ButtonIcon);
    ui->x_0_motta->setIconSize(pix.rect().size());
    ui->x_0_motta->setFixedSize(pix.rect().size());

//Objekter - Index nummer -------------------------------------------------------------------------------------------------------------------

    ui->stackedWidget->insertWidget(6, &_Vaatlakk);                     //setter inn vaatlakkering.cpp i index 6, side 7
    ui->stackedWidget->insertWidget(7, &_Pulverlakk);                   //setter inn pulverlakkering.cpp i index 7, side 8
    ui->stackedWidget->insertWidget(8, &_Offshorelakk);                 //setter inn offshore_lakkering.cpp i index 8, side 9
    ui->stackedWidget->insertWidget(9, &_Eloks);                        //Setter inn eloksering.cpp i index 9, side 10
    ui->stackedWidget->insertWidget(10, &_Kromater);                    //Setter inn kromatering.cpp i index 10, side 11
    ui->stackedWidget->insertWidget(11, &_Elzink);                      //Setter inn elzink.cpp i index 11, side 12
    ui->stackedWidget->insertWidget(12, &_Hardforkrom);                 //Setter inn hardforkromming.cpp i index 12, side 13
    ui->stackedWidget->insertWidget(13, &_ElektroFornikling);           //Setter inn elektrolytisk_fornikling.cpp i index 13, side 14
    ui->stackedWidget->insertWidget(14, &_Fosfat);                      //Setter inn fosfatering.cpp i index 14, side 13
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

//Signaler for tilbakeknappene mellom cpp.filene -------------------------------------------------------------------------------------------------------------------
    //lakkering
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

mottar::~mottar()
{
    delete ui;
}

// lakkeringsvindu-------------------------------------------------------------------------------
void mottar::on_lakkering_clicked()                                     //Lakkering - knapp
{
    ui->stackedWidget->setCurrentIndex(1);  //index 1 = side 2
}
    void mottar::on_vaatlakkering_clicked()                             //Våtlakkering - knapp
    {
        ui->stackedWidget->setCurrentIndex(6);
    }
    void mottar::tilbake_0_vaat_bedrifter()                             //tilbakeknapp i index 0 - våtlakkering
    {
        ui->stackedWidget->setCurrentIndex(1);
    }

    void mottar::on_pulverlakkering_clicked()                           //Pulverlakkering - knapp
    {
        ui->stackedWidget->setCurrentIndex(7);
    }
    void mottar::tilbake_0_pulver_bedrifter()                           //tilbakeknapp i index 0 - pulverlakkering
    {
        ui->stackedWidget->setCurrentIndex(1);
    }

    void mottar::on_offshore_lakkering_clicked()                        //Offshore lakkering - knapp
    {
        ui->stackedWidget->setCurrentIndex(8);
    }
    void mottar::tilbake_0_offshore_bedrifter()                         //tilbakeknapp i index 0 - offshore lakkering
    {
        ui->stackedWidget->setCurrentIndex(1);
    }

//elektrokjemisk metode vindu-------------------------------------------------------------------------------
void mottar::on_elektrokjemisk_metode_clicked()                         //Elektrokjemisk_metode - knapp
{
    ui->stackedWidget->setCurrentIndex(2);
}
    void mottar::on_eloksering_alu_clicked()                            //Eloksering - knapp
    {
        ui->stackedWidget->setCurrentIndex(9);
    }
    void mottar::tilbake_0_eloksering_bedrifter()                       //tilbakeknapp i index 0 - eloksering
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void mottar::on_kromatering_clicked()                               //Kromatering - knapp
    {
        ui->stackedWidget->setCurrentIndex(10);
    }
    void mottar::tilbake_0_kromatering_bedrifter()                      //tilbakeknapp i index 0 - kromatering
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void mottar::on_elzink_clicked()                                    //Elzink - knapp
    {
        ui->stackedWidget->setCurrentIndex(11);
    }
    void mottar::tilbake_0_elzink_bedrifter()                           //tilbakeknapp i index 0 - elzink
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void mottar::on_hardforkromming_clicked()                           //Hardforkromming - knapp
    {
        ui->stackedWidget->setCurrentIndex(12);
    }
    void mottar::tilbake_0_hardforkromming_bedrifter()                  //tilbakeknapp i index 0 - hardforkromming
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

    void mottar::on_elektrolytisk_fornikling_clicked()                  //Elektrolytisk_fornikling - knapp
    {
        ui->stackedWidget->setCurrentIndex(13);
    }
    void mottar::tilbake_0_elektrolytisk_fornikling_bedrifter()         //tilbakeknapp i index 0 - Elektrolytisk_fornikling
    {
        ui->stackedWidget->setCurrentIndex(2);
    }

//kjemisk metode vindu-------------------------------------------------------------------------------
void mottar::on_kjemisk_metode_clicked()                                //Kjemisk_metode - knapp
{
    ui->stackedWidget->setCurrentIndex(3);
}
    void mottar::on_fosfatering_clicked()                               //Fosfatering - knapp
    {
        ui->stackedWidget->setCurrentIndex(14);
    }
    void mottar::tilbake_0_fosfatering_bedrifter()                      //tilbakeknapp i index 0 - fosfatering
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

    void mottar::on_galvanisering_varmforsinking_clicked()              //Galvanisering - knapp
    {
        ui->stackedWidget->setCurrentIndex(15);
    }
    void mottar::tilbake_0_galvanisering_bedrifter()                    //tilbakeknapp i index 0 - galvanisering
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

    void mottar::on_syrevask_clicked()                                  //Syrevask - knapp
    {
        ui->stackedWidget->setCurrentIndex(16);
    }
    void mottar::tilbake_0_syrevask_bedrifter()                         //tilbakeknapp i index 0 - Syrevask
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

    void mottar::on_brunering_clicked()                                 //Brunering - knapp
    {
        ui->stackedWidget->setCurrentIndex(17);
    }
    void mottar::tilbake_0_brunering_bedrifter()                        //tilbakeknapp i index 0 - brunering
    {
        ui->stackedWidget->setCurrentIndex(3);
    }

//herding vindu-------------------------------------------------------------------------------
void mottar::on_herding_clicked()                                       //Herding
{
    ui->stackedWidget->setCurrentIndex(4);
}
    void mottar::on_herd_herding_clicked()                              //Herding - knapp
    {
        ui->stackedWidget->setCurrentIndex(18);
    }
    void mottar::tilbake_0_herding_bedrifter()                          //tilbakeknapp i index 0 - herding
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

    void mottar::on_sett_herding_clicked()                              //Sett_herding - knapp
    {
        ui->stackedWidget->setCurrentIndex(19);
    }
    void mottar::tilbake_0_sett_herding_bedrifter()                     //tilbakeknapp i index 0 - sett_herding
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

    void mottar::on_nitrering_clicked()                                 //Nitrering - knapp
    {
        ui->stackedWidget->setCurrentIndex(20);
    }
    void mottar::tilbake_0_nitrering_bedrifter()                        //tilbakeknapp i index 0 - nitrering
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

    void mottar::on_plasma_nitrering_clicked()                          //Plasmanitrering - knapp
    {
        ui->stackedWidget->setCurrentIndex(21);
    }
    void mottar::tilbake_0_plasma_nitrering_bedrifter()                 //tilbakeknapp i index 0 - Plasmanitrering
    {
        ui->stackedWidget->setCurrentIndex(4);
    }

//annet vindu-------------------------------------------------------------------------------
void mottar::on_annet_clicked()                                         //Annet - knapp
{
    ui->stackedWidget->setCurrentIndex(5);  //index 1 = side 6
}
    void mottar::on_sliping_clicked()                                   //Sliping - knapp
    {
        ui->stackedWidget->setCurrentIndex(22);
    }
    void mottar::tilbake_0_sliping_bedrifter()                          //tilbakeknapp i index 0 - sliping
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_traadskjaering_clicked()                            //Trådskjæring - knapp
    {
        ui->stackedWidget->setCurrentIndex(23);
    }
    void mottar::tilbake_0_traadskjaering_bedrifter()                   //tilbakeknapp i index 0 - trådskjæring
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_glass_blaasing_clicked()                            //Glass_blåsing - knapp
    {
        ui->stackedWidget->setCurrentIndex(24);
    }
    void mottar::tilbake_0_glass_blaasing_bedrifter()                   //tilbakeknapp i index 0 - glass_blåsing
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_sandblaasing_clicked()                              //Sandblåsing - knapp
    {
        ui->stackedWidget->setCurrentIndex(25);
    }
    void mottar::tilbake_0_sandblaasing_bedrifter()                     //tilbakeknapp i index 0 - sandblåsing
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_polering_clicked()                                  //Polering - knapp
    {
        ui->stackedWidget->setCurrentIndex(26);
    }
    void mottar::tilbake_0_polering_bedrifter()                         //tilbakeknapp i index 0 - polering
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_elektrolytisk_polering_clicked()                    //Elektrolytisk_polering - knapp
    {
        ui->stackedWidget->setCurrentIndex(27);
    }
    void mottar::tilbake_0_elektrolytisk_polering_bedrifter()           //tilbakeknapp i index 0 - elektrolytisk_polering
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_forsolving_gullbelegging_clicked()                  //Forsølving_Gullbelegging - knapp
    {
        ui->stackedWidget->setCurrentIndex(28);
    }
    void mottar::tilbake_0_forsolving_bedrifter()                       //tilbakeknapp i index 0 - forsølving_gullbelegging
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_gullbelegging_clicked()                             //Gullbelegging - knapp
    {
        ui->stackedWidget->setCurrentIndex(29);
    }
    void mottar::tilbake_0_gullbelegging_bedrifter()                    //tilbakeknapp i index 0 - gullbelegging
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

    void mottar::on_fortinning_clicked()                                //Fortinning - knapp
    {
        ui->stackedWidget->setCurrentIndex(30);
    }
    void mottar::tilbake_0_fortinning_bedrifter()                       //tilbakeknapp i index 0 - fortinning
    {
        ui->stackedWidget->setCurrentIndex(5);
    }

// Tilleggs widgets: --------------------------------------------------------------------------------------------------------
void mottar::on_tilbake_0_motta_clicked()                               //tilbakeknapp i motta.cpp --> Hjem/logg inn - SIGNAL
{
    emit tilbake_0_motta_clicked();         // avslutter og tilbake til hjem (index 0, side 1 i main).
}

void mottar::on_x_0_motta_clicked()                                     //Avslutt --> Hjem/Logg inn - SIGNAL
{
    emit x_0_motta_clicked();               //ett hakk tilbake (index 0, side 1 i sender) - Destinasjon: SEND og MOTTA
}

void mottar::on_tilbake_1_lakk_motta_clicked()                          //Tilbakeknapp i lakkeringvindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void mottar::on_tilbake_2_elkj_motta_clicked()                          //Tilbakeknapp i elektrokjemisk_metode vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void mottar::on_tilbake_3_kj_motta_clicked()                            //Tilbakeknapp i kjemisk_metode vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void mottar::on_tilbake_4_herding_motta_clicked()                       //Tilbakeknapp i herding vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}

void mottar::on_tilbake_5_annet_motta_clicked()                         //tilbakeknapp i annet vindu
{
    ui->stackedWidget->setCurrentIndex(0);  //tilbake til index 0, side 1
}
