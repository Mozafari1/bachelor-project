#ifndef OFFSHORE_LAKKERING_SEND_H
#define OFFSHORE_LAKKERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class offshore_lakkering_send;
}

class offshore_lakkering_send : public QWidget
{
    Q_OBJECT

public:
    explicit offshore_lakkering_send(QWidget *parent = nullptr);
    ~offshore_lakkering_send();

private slots:
    void on_nct_clicked();

    void on_tilbake_0_offshore_nct_clicked();
    void on_tilbake_0_offshore_bedrifter_clicked();

private:
    Ui::offshore_lakkering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_offshore_bedrifter_clicked();
};

#endif // OFFSHORE_LAKKERING_SEND_H
