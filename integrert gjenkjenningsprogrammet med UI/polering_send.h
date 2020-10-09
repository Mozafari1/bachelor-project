#ifndef POLERING_SEND_H
#define POLERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class polering_send;
}

class polering_send : public QWidget
{
    Q_OBJECT

public:
    explicit polering_send(QWidget *parent = nullptr);
    ~polering_send();

private slots:
    void on_tilbake_0_polering_bedrifter_clicked();

    void on_tronrud_clicked();
    void on_tilbake_1_polering_tronrud_clicked();

    void on_honefoss_clicked();
    void on_tilbake_2_polering_honefoss_clicked();

private:
    Ui::polering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_polering_bedrifter_clicked();
};

#endif // POLERING_SEND_H
