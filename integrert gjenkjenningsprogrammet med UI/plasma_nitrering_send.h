#ifndef PLASMA_NITRERING_SEND_H
#define PLASMA_NITRERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class plasma_nitrering_send;
}

class plasma_nitrering_send : public QWidget
{
    Q_OBJECT

public:
    explicit plasma_nitrering_send(QWidget *parent = nullptr);
    ~plasma_nitrering_send();

private slots:
    void on_tilbake_0_plasma_nitrering_bedrifter_clicked();

    void on_bodycote_clicked();
    void on_tilbake_1_plasma_nitrering_bodycote_clicked();

private:
    Ui::plasma_nitrering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_plasma_nitrering_bedrifter_clicked();
};

#endif // PLASMA_NITRERING_SEND_H
