#ifndef SYREVASK_SEND_H
#define SYREVASK_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class syrevask_send;
}

class syrevask_send : public QWidget
{
    Q_OBJECT

public:
    explicit syrevask_send(QWidget *parent = nullptr);
    ~syrevask_send();

private slots:
    void on_tilbake_0_syrevask_bedrifter_clicked();

    void on_tl_clicked();
    void on_tilbake_1_syrevask_tl_clicked();

    void on_gundersen_galvano_vestby_clicked();
    void on_tilbake_2_syrevask_gundersen_vestby_clicked();

    void on_eiker_overflateteknikk_clicked();
    void on_tilbake_3_syrevask_eiker_clicked();

    void on_nct_clicked();
    void on_tilbake_4_syrevask_nct_clicked();

private:
    Ui::syrevask_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_syrevask_bedrifter_clicked();
};

#endif // SYREVASK_SEND_H
