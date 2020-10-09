#ifndef SETT_HERDING_SEND_H
#define SETT_HERDING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class sett_herding_send;
}

class sett_herding_send : public QWidget
{
    Q_OBJECT

public:
    explicit sett_herding_send(QWidget *parent = nullptr);
    ~sett_herding_send();

private slots:
    void on_tilbake_0_sett_herding_bedrifter_clicked();

    void on_thune_produkter_raufoss_clicked();
    void on_tilbake_1_sett_herding_thune_clicked();

    void on_bodycote_clicked();
    void on_tilbake_2_sett_herding_bodycote_clicked();

private:
    Ui::sett_herding_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_sett_herding_bedrifter_clicked();
};

#endif // SETT_HERDING_SEND_H
