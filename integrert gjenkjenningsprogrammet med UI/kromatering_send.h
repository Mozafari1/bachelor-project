#ifndef KROMATERING_SEND_H
#define KROMATERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class kromatering_send;
}

class kromatering_send : public QWidget
{
    Q_OBJECT

public:
    explicit kromatering_send(QWidget *parent = nullptr);
    ~kromatering_send();

private slots:
    void on_tilbake_0_kromatering_bedrifter_clicked();

    void on_eloksal_clicked();
    void on_tilbake_1_eloksal_clicked();

    void on_gundersen_galvano_clicked();
    void on_tilbake_2_gundersen_clicked();

    void on_gundersen_galvano_vestby_clicked();
    void on_tilbake_3_gundersen_vestby_clicked();

    void on_eiker_overflateteknikk_clicked();
    void on_tilbake_4_eiker_clicked();

    void on_nct_clicked();
    void on_tilbake_5_nct_clicked();

private:
    Ui::kromatering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_kromatering_bedrifter_clicked();
};

#endif // KROMATERING_SEND_H
