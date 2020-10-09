#ifndef VAATLAKKERING_SEND_H
#define VAATLAKKERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>

namespace Ui {
class vaatlakkering_send;
}

class vaatlakkering_send : public QWidget
{
    Q_OBJECT

public:
    explicit vaatlakkering_send(QWidget *parent = nullptr);
    ~vaatlakkering_send();

private slots:
    void on_tilbake_0_vaat_bedrifter_clicked();
    void on_bilskadesenteret_ringerike_clicked();
    void on_lyseng_billakkering_clicked();
    void on_tilbake_1_vaat_bilskade_clicked();
    void on_tilbake_2_vaat_lyseng_clicked();

private:
    Ui::vaatlakkering_send *ui;
    class gjenkjenning_send _GjenkjennSend;

signals:
    void tilbake_0_vaat_bedrifter_clicked();
};

#endif // VAATLAKKERING_SEND_H
