#ifndef SENDER_H
#define SENDER_H

#include <QWidget>

#include <vaatlakkering_send.h>
#include <pulverlakkering_send.h>
#include <offshore_lakkering_send.h>
#include <eloksering_send.h>
#include <kromatering_send.h>
#include <elzink_send.h>
#include <hardforkromming_send.h>
#include <elektrolytisk_fornikling_send.h>
#include <fosfatering_send.h>
#include <galvanisering_send.h>
#include <syrevask_send.h>
#include <brunering_send.h>
#include <herding_send.h>
#include <sett_herding_send.h>
#include <nitrering_send.h>
#include <plasma_nitrering_send.h>
#include <sliping_send.h>
#include <traadskjaering_send.h>
#include <glass_blaasing_send.h>
#include <sandblaasing_send.h>
#include <polering_send.h>
#include <elektrolytisk_polering_send.h>
#include <forsolving_send.h>
#include <gullbelegging_send.h>
#include <fortinning_send.h>


namespace Ui {
class sender;
}

class sender : public QWidget
{
    Q_OBJECT

public:
    explicit sender(QWidget *parent = nullptr);
    ~sender();

private slots:
    //index 0, side 1:
    void on_x_0_send_clicked();                                     //avslutt (tilbake til hjem/logg inn) - mainwindow.cpp - KNAPP (x)
    void on_tilbake_0_send_clicked();

    void on_lakkering_clicked();                                    //Lakkering - knapp
    void on_elektrokjemisk_metode_clicked();                        //Elektrokjemisk_metode - knapp
    void on_kjemisk_metode_clicked();                               //Kjemisk_metode - knapp
    void on_herding_clicked();                                      //Herding - knapp
    void on_annet_clicked();                                        //Annet - knapp

    void on_tilbake_1_lakk_send_clicked();                          //tilbakeknapp i lakkeringvindu
    void on_tilbake_2_elkj_send_clicked();                          //tilbakeknapp i elektrokjemisk_metode vindu
    void on_tilbake_3_kj_send_clicked();                            //tilbakeknapp i kjemisk_metode vindu
    void on_tilbake_4_herd_send_clicked();                          //tilbakeknapp i herding vindu
    void on_tilbake_5_annet_send_clicked();                         //tilbakeknapp i annet vinudu

//Lakkeringsvindu---------------------------------------------------------------------------------------------------------------------
    void on_vaatlakkering_clicked();                                //Våtlakkering
    void tilbake_0_vaat_bedrifter();                                //tilbake fra våtlakkering til lakkering

    void on_pulverlakkering_clicked();                              //pulverlakkering
    void tilbake_0_pulver_bedrifter();                              //tilbake fra pulverlakkering til lakkering

    void on_offshore_lakkering_clicked();                           //Offshore lakkering
    void tilbake_0_offshore_bedrifter();                            //tilbake fra offshore lakkering til lakkering

//elektrokjemisk metode vindu---------------------------------------------------------------------------------------------------------
    void on_eloksering_alu_clicked();                               //eloksering
    void tilbake_0_eloksering_bedrifter();                          //tilbake fra eloksering til elektrokjemisk metode

    void on_kromatering_clicked();                                  //Kromatering
    void tilbake_0_kromatering_bedrifter();                         //tilbake fra kromatering til elektrokjemisk metode

    void on_elzink_clicked();                                       //Elektrolytisk forsinking (ELZINK)
    void tilbake_0_elzink_bedrifter();                              //tilbake fra elzink til elektrokjemisk metode

    void on_hardforkromming_clicked();                              //Hardforkromming
    void tilbake_0_hardforkromming_bedrifter();                     //tilbake fra hardforkromming til elektrokjemisk metode

    void on_elektrolytisk_fornikling_clicked();                     //Elektrolytisk fornikling
    void tilbake_0_elektrolytisk_fornikling_bedrifter();            //tilbake fra elektrolytisk fornikling til elektrokjemisk metode

//kjemisk metode vindu----------------------------------------------------------------------------------------------------------------

    void on_fosfatering_clicked();                                  //fosfatering
    void tilbake_0_fosfatering_bedrifter();                         //tilbake fra fosfatering til kjemisk metode

    void on_galvanisering_varmforsinking_clicked();                 //galvanisering/varmforsinking
    void tilbake_0_galvanisering_bedrifter();                       //tilbake fra galvanisering/varmforsinking til kjemisk metode

    void on_syrevask_clicked();                                     //syrevask
    void tilbake_0_syrevask_bedrifter();                            //tilbake fra syrevask til kjemisk metode

    void on_brunering_clicked();                                    //brunering
    void tilbake_0_brunering_bedrifter();                           //tilbake fra brunering til kjemisk metode

//herding vindu----------------------------------------------------------------------------------------------------------------

    void on_herd_herding_clicked();                                 //herding
    void tilbake_0_herding_bedrifter();                             //tilbake fra herding til herding

    void on_sett_herding_clicked();                                 //sett-herding
    void tilbake_0_sett_herding_bedrifter();                        //tilbake fra sett-herding til herding

    void on_nitrering_clicked();                                    //nitrering
    void tilbake_0_nitrering_bedrifter();                           //tilbake fra nitrering til herding

    void on_plasma_nitrering_clicked();                             //plasma nitrering
    void tilbake_0_plasma_nitrering_bedrifter();                    //tilbake fra plasma nitrering til herding

//annet vindu----------------------------------------------------------------------------------------------------------------

    void on_sliping_clicked();                                      //sliping
    void tilbake_0_sliping_bedrifter();                             //tilbake fra sliping til annet

    void on_traadskjaering_clicked();                               //trådskjæring
    void tilbake_0_traadskjaering_bedrifter();                      //tilbake fra trådskjæring til annet

    void on_glass_blaasing_clicked();                               //glass blåsing
    void tilbake_0_glass_blaasing_bedrifter();                      //tilbake fra glass blåsing til annet

    void on_sandblaasing_clicked();                                 //sandblåsing
    void tilbake_0_sandblaasing_bedrifter();                        //tilbake fra sandblåsing til annet

    void on_polering_clicked();                                     //polering
    void tilbake_0_polering_bedrifter();                            //tilbake fra polering til annet

    void on_elektrolytisk_polering_clicked();                       //elektrolytisk polering
    void tilbake_0_elektrolytisk_polering_bedrifter();              //tilbake fra elektrolytisk polering til annet

    void on_forsolving_gullbelegging_clicked();                     //forsølving
    void tilbake_0_forsolving_bedrifter();                          //tilbake fra forsølving til annet

    void on_gullbelegging_clicked();                                //gullbelegging
    void tilbake_0_gullbelegging_bedrifter();                       //tilbake fra gullbelegging til annet

    void on_fortinning_clicked();                                   //fortinning
    void tilbake_0_fortinning_bedrifter();                          //tilbake fra fortinning til annet

private:
    Ui::sender *ui;
    class vaatlakkering_send _Vaatlakk;                             //objekt for våtlakkering
    class pulverlakkering_send _Pulverlakk;                         //objekt for pulverlakkering
    class offshore_lakkering_send _Offshorelakk;                    //objekt for offshore lakkering

    class eloksering_send _Eloks;                                   //objekt for eloksering (elektrokjemisk metode)
    class kromatering_send _Kromater;                               //objekt for kromatering (elektrokjemisk metode)
    class elzink_send _Elzink;                                      //objekt for elzink (elektrokjemisk metode)
    class hardforkromming_send _Hardforkrom;                        //objekt for hardforkromming (elektrokjemisk metode)
    class elektrolytisk_fornikling_send _ElektroFornikling;         //objekt for elektrolytisk fornikling (elektrokjemisk metode)

    class fosfatering_send _Fosfat;                                 //objekt for fosfatering (kjemisk metode)
    class galvanisering_send _Galv;                                 //objekt for galvanisering/varmforsinking (kjemisk metode)
    class syrevask_send _Syre;                                      //objekt for syrevask (kjemisk metode)
    class brunering_send _Brun;                                     //objekt for brunering (kjemisk metode)

    class herding_send _Herd;                                       //objekt for herding (herding)
    class sett_herding_send _SettHerd;                              //objekt for sett-herding (herding)
    class nitrering_send _Nitrer;                                   //objekt for nitrering (herding)
    class plasma_nitrering_send _PlasmaNitrer;                      //objekt for plasma nitrering (herding)

    class sliping_send _Sliping;                                    //objekt for sliping (Annet)
    class traadskjaering_send _Traad;                               //objekt for trådskjæring (Annet)
    class glass_blaasing_send _Glass;                               //objekt for glass blåsing (Annet)
    class sandblaasing_send _Sand;                                  //objekt for sandblåsing (Annet)
    class polering_send _Poler;                                     //objekt for polering (Annet)
    class elektrolytisk_polering_send _ElektroPolering;             //objekt for elektrolytisk polering (Annet)
    class forsolving_send _Forsolv;                                 //objekt for forsølving/Gullbelegging (Annet)
    class gullbelegging_send _Gull;                                 //objekt for gullbelegging (Annet)
    class fortinning_send _Fortinning;                              //objekt for fortinning (Annet)

signals:
    void x_0_send_clicked();                                        //avslutt (tilbake til hjem/logg inn) - mainwindow.cpp - SIGNAL
    void tilbake_0_send_clicked();                                  //Tilbakeknapp i send.cpp filen - SIGNAL
};

#endif // SENDER_H
