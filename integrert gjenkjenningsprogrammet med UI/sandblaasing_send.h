#ifndef SANDBLAASING_SEND_H
#define SANDBLAASING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class sandblaasing_send;
}

class sandblaasing_send : public QWidget
{
    Q_OBJECT

public:
    explicit sandblaasing_send(QWidget *parent = nullptr);
    ~sandblaasing_send();

private slots:
    void on_tilbake_0_sandblaasing_bedrifter_clicked();

    void on_bilskadesenteret_clicked();
    void on_tilbake_1_sandblaasing_bilskadesenteret_clicked();

    void on_nct_clicked();
    void on_tilbake_2_sandblaasing_nct_clicked();

private:
    Ui::sandblaasing_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_sandblaasing_bedrifter_clicked();
};

#endif // SANDBLAASING_SEND_H
