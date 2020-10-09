#ifndef GLASS_BLAASING_SEND_H
#define GLASS_BLAASING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class glass_blaasing_send;
}

class glass_blaasing_send : public QWidget
{
    Q_OBJECT

public:
    explicit glass_blaasing_send(QWidget *parent = nullptr);
    ~glass_blaasing_send();

private slots:
    void on_tilbake_0_glass_blaasing_bedrifter_clicked();

    void on_tronrud_clicked();
    void on_tilbake_1_glass_blaasing_tronrud_clicked();

private:
    Ui::glass_blaasing_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_glass_blaasing_bedrifter_clicked();
};

#endif // GLASS_BLAASING_SEND_H
