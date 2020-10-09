#ifndef ELEKTROLYTISK_POLERING_SEND_H
#define ELEKTROLYTISK_POLERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class elektrolytisk_polering_send;
}

class elektrolytisk_polering_send : public QWidget
{
    Q_OBJECT

public:
    explicit elektrolytisk_polering_send(QWidget *parent = nullptr);
    ~elektrolytisk_polering_send();

private slots:
    void on_tilbake_0_elektrolytisk_polering__bedrifter_clicked();

    void on_gundersen_vestby_clicked();
    void on_tilbake_1_elektrolytisk_polering_gundersen_clicked();

private:
    Ui::elektrolytisk_polering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_elektrolytisk_polering_bedrifter_clicked();
};

#endif // ELEKTROLYTISK_POLERING_SEND_H
