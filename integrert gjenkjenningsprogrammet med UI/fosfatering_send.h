#ifndef FOSFATERING_SEND_H
#define FOSFATERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class fosfatering_send;
}

class fosfatering_send : public QWidget
{
    Q_OBJECT

public:
    explicit fosfatering_send(QWidget *parent = nullptr);
    ~fosfatering_send();

private slots:
    void on_tilbake_0_fosfatering_bedrifter_clicked();

    void on_nct_clicked();
    void on_tilbake_1_fosfatering_nct_clicked();

    void on_gundersen_clicked();
    void on_tilbake_2_fosfatering_gundersen_clicked();

private:
    Ui::fosfatering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_fosfatering_bedrifter_clicked();
};

#endif // FOSFATERING_SEND_H
