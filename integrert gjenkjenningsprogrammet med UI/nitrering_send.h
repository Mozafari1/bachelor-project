#ifndef NITRERING_SEND_H
#define NITRERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class nitrering_send;
}

class nitrering_send : public QWidget
{
    Q_OBJECT

public:
    explicit nitrering_send(QWidget *parent = nullptr);
    ~nitrering_send();

private slots:
    void on_tilbake_0_nitrering_bedrifter_clicked();

    void on_thune_produkter_raufoss_clicked();
    void on_tilbake_1_nitrering_thune_clicked();

    void on_bodycote_clicked();
    void on_tilbake_2_nitrering_bodycote_clicked();

private:
    Ui::nitrering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_nitrering_bedrifter_clicked();
};

#endif // NITRERING_SEND_H
