#ifndef MOTTAR_H
#define MOTTAR_H

#include <QWidget>

#include <vaatlakkering_motta.h>
#include <pulverlakkering_motta.h>
#include <offshore_lakkering_motta.h>
#include <eloksering_motta.h>
#include <kromatering_motta.h>
#include <elzink_motta.h>
#include <hardforkromming_motta.h>
#include <elektrolytisk_fornikling_motta.h>
#include <fosfatering_motta.h>
#include <galvanisering_motta.h>
#include <syrevask_motta.h>
#include <brunering_motta.h>
#include <herding_motta.h>
#include <sett_herding_motta.h>
#include <nitrering_motta.h>
#include <plasma_nitrering_motta.h>
#include <sliping_motta.h>
#include <traadskjaering_motta.h>
#include <glass_blaasing_motta.h>
#include <sandblaasing_motta.h>
#include <polering_motta.h>
#include <elektrolytisk_polering_motta.h>
#include <forsolving_motta.h>
#include <gullbelegging_motta.h>
#include <fortinning_motta.h>

namespace Ui {
class mottar;
}

class mottar : public QWidget
{
    Q_OBJECT

public:
    explicit mottar(QWidget *parent = nullptr);
    ~mottar();

private slots:
    void on_x_0_motta_clicked();                                    //Avslutt (tilbake til hjem/logg inn) - mainwindow.cpp - KNAPP (hjem ikon)
    void on_tilbake_0_motta_clicked();                              //Tilbakeknapp i motta.cpp - knapp

    void on_lakkering_clicked();                                    //Lakkering - knapp
    void on_elektrokjemisk_metode_clicked();                        //Elektrokjemisk_metode - knapp
    void on_kjemisk_metode_clicked();                               //Kjemisk_metode - knapp
    void on_herding_clicked();                                      //Herding - knapp
    void on_annet_clicked();                                        //Annet - annet

    void on_tilbake_1_lakk_motta_clicked();                         //tilbakeknapp i lakkeringvindu
    void on_tilbake_2_elkj_motta_clicked();                         //tilbakeknapp i elektrokjemisk_metode vindu
    void on_tilbake_3_kj_motta_clicked();                           //tilbakeknapp i kjemisk_metode vindu
    void on_tilbake_4_herding_motta_clicked();                      //tilbakeknapp i herding vindu
    void on_tilbake_5_annet_motta_clicked();                        //tilbakeknapp i annet vindu

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
    Ui::mottar *ui;
    class vaatlakkering_motta _Vaatlakk;                            //objekt for våtlakkering
    class pulverlakkering_motta _Pulverlakk;                        //objekt for pulverlakkering
    class offshore_lakkering_motta _Offshorelakk;                   //objekt for offshore lakkering

    class eloksering_motta _Eloks;                                  //objekt for eloksering (elektrokjemisk metode)
    class kromatering_motta _Kromater;                              //objekt for kromatering (elektrokjemisk metode)
    class elzink_motta _Elzink;                                     //objekt for Elektrolytisk forsinking/elzink (elektrokjemisk metode)
    class hardforkromming_motta _Hardforkrom;                       //objekt for hardforkromming (elektrokjemisk metode)
    class elektrolytisk_fornikling_motta _ElektroFornikling;        //objekt for elektrolytisk fornikling (elektrokjemisk metode)

    class fosfatering_motta _Fosfat;                                //objekt for fosfatering (kjemisk metode)
    class galvanisering_motta _Galv;                                //objekt for galvanisering/varmforsinking (kjemisk metode)
    class syrevask_motta _Syre;                                     //objekt for syrevask (kjemisk metode)
    class brunering_motta _Brun;                                    //objekt for brunering (kjemisk metode)

    class herding_motta _Herd;                                      //objekt for herding (herding)
    class sett_herding_motta _SettHerd;                             //objekt for sett-herding (herding)
    class nitrering_motta _Nitrer;                                  //objekt for nitrering (herding)
    class plasma_nitrering_motta _PlasmaNitrer;                     //objekt for plasma nitrering (herding)

    class sliping_motta _Sliping;                                   //objekt for sliping (Annet)
    class traadskjaering_motta _Traad;                              //objekt for trådskjæring (Annet)
    class glass_blaasing_motta _Glass;                              //objekt for glass blåsing (Annet)
    class sandblaasing_motta _Sand;                                 //objekt for sandblåsing (Annet)
    class polering_motta _Poler;                                    //objekt for polering (Annet)
    class elektrolytisk_polering_motta _ElektroPolering;            //objekt for elektrolytisk polering (Annet)
    class forsolving_motta _Forsolv;                                //objekt for forsølving/Gullbelegging (Annet)
    class gullbelegging_motta _Gull;                                //objekt for gullbelegging (Annet)
    class fortinning_motta _Fortinning;                             //objekt for fortinning (Annet)

signals:
    void x_0_motta_clicked();                                       //Avslutt (tilbake til hjem/logg inn) mainwindow.cpp - SIGNAL
    void tilbake_0_motta_clicked();                                 //Tilbakeknapp i motta.cpp - SIGNAL

};

#endif // MOTTAR_H
