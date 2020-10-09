#ifndef HERDING_SEND_H
#define HERDING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class herding_send;
}

class herding_send : public QWidget
{
    Q_OBJECT

public:
    explicit herding_send(QWidget *parent = nullptr);
    ~herding_send();

private slots:
    void on_tilbake_0_herding_bedrifter_clicked();

    void on_thune_produkter_raufoss_clicked();
    void on_tilbake_1_herding_thune_clicked();

    void on_bodycote_clicked();
    void on_tilbake_2_herding_bodycote_clicked();

private:
    Ui::herding_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_herding_bedrifter_clicked();
};

#endif // HERDING_SEND_H
