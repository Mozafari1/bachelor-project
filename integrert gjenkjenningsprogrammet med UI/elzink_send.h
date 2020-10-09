#ifndef ELZINK_SEND_H
#define ELZINK_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class elzink_send;
}

class elzink_send : public QWidget
{
    Q_OBJECT

public:
    explicit elzink_send(QWidget *parent = nullptr);
    ~elzink_send();

private slots:
    void on_gundersen_galvano_vestby_clicked();

    void on_eiker_overflateteknikk_clicked();

    void on_nct_clicked();

    void on_tilbake_1_elzink_gundersen_vestby_clicked();

    void on_tilbake_2_elzink_eiker_clicked();

    void on_tilbake_3_elzink_honefoss_clicked();


    void on_tilbake_0_elzink_bedrifter_clicked();

private:
    Ui::elzink_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_elzink_bedrifter_clicked();
};

#endif // ELZINK_SEND_H
