#ifndef BRUNERING_SEND_H
#define BRUNERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class brunering_send;
}

class brunering_send : public QWidget
{
    Q_OBJECT

public:
    explicit brunering_send(QWidget *parent = nullptr);
    ~brunering_send();

private slots:
    void on_tilbake_0_brunering_bedrifter_clicked();

    void on_borsemakern_clicked();
    void on_tilbake_1_brunering_borsemakern_clicked();

    void on_eiker_clicked();
    void on_tilbake_2_brunering_eiker_clicked();

private:
    Ui::brunering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_brunering_bedrifter_clicked();
};

#endif // BRUNERING_SEND_H
