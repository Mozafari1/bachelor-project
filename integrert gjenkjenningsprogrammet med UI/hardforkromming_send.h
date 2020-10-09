#ifndef HARDFORKROMMING_SEND_H
#define HARDFORKROMMING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class hardforkromming_send;
}

class hardforkromming_send : public QWidget
{
    Q_OBJECT

public:
    explicit hardforkromming_send(QWidget *parent = nullptr);
    ~hardforkromming_send();

private slots:
    void on_tilbake_0_harforkromming_bedrifter_clicked();

    void on_norsk_everit_clicked();
    void on_tilbake_1_hardforkromming_norsk_everit_clicked();

    void on_eiker_clicked();
    void on_tilbake_2_hardforkromming_eiker_clicked();



private:
    Ui::hardforkromming_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_hardforkromming_bedrifter_clicked();
};

#endif // HARDFORKROMMING_SEND_H
