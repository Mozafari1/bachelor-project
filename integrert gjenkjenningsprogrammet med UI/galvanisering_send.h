#ifndef GALVANISERING_SEND_H
#define GALVANISERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class galvanisering_send;
}

class galvanisering_send : public QWidget
{
    Q_OBJECT

public:
    explicit galvanisering_send(QWidget *parent = nullptr);
    ~galvanisering_send();

private slots:
    void on_tilbake_0_galvanisering_bedrifter_clicked();

    void on_br_berntsen_clicked();
    void on_tilbake_1_galvanisering_br_berntsen_clicked();

    void on_gundersen_galvano_clicked();
    void on_tilbake_2_galvanisering_gundersen_clicked();

    void on_duozink_clicked();
    void on_tilbake_3_galvanisering_duozink_clicked();



private:
    Ui::galvanisering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_galvanisering_bedrifter_clicked();
};

#endif // GALVANISERING_SEND_H
