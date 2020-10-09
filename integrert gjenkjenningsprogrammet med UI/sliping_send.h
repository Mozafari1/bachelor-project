#ifndef SLIPING_SEND_H
#define SLIPING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class sliping_send;
}

class sliping_send : public QWidget
{
    Q_OBJECT

public:
    explicit sliping_send(QWidget *parent = nullptr);
    ~sliping_send();

private slots:
    void on_tilbake_0_sliping_bedrifter_clicked();

    void on_thune_clicked();
    void on_tilbake_1_sliping_thune_clicked();

private:
    Ui::sliping_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_sliping_bedrifter_clicked();
};

#endif // SLIPING_SEND_H
