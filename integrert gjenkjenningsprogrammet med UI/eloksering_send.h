#ifndef ELOKSERING_SEND_H
#define ELOKSERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class eloksering_send;
}

class eloksering_send : public QWidget
{
    Q_OBJECT

public:
    explicit eloksering_send(QWidget *parent = nullptr);
    ~eloksering_send();

private slots:


    void on_tilbake_0_eloksering_bedrifter_clicked();

    void on_eloksal_clicked();
    void on_tilbake_1_eloksal_clicked();

    void on_gundersen_galvano_clicked();
    void on_tilbake_2_gundersen_clicked();

    void on_hydal_alu_profiler_clicked();
    void on_tilbake_3_hydal_alu_clicked();

    void on_teloks_clicked();
    void on_tilbake_4_teloks_clicked();

    void on_nct_clicked();
    void on_tilbake_5_nct_clicked();

private:
    Ui::eloksering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_eloksering_bedrifter_clicked();
};

#endif // ELOKSERING_SEND_H
