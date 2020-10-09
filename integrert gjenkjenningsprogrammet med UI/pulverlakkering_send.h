#ifndef PULVERLAKKERING_SEND_H
#define PULVERLAKKERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class pulverlakkering_send;
}

class pulverlakkering_send : public QWidget
{
    Q_OBJECT

public:
    explicit pulverlakkering_send(QWidget *parent = nullptr);
    ~pulverlakkering_send();

private slots:
    void on_tilbake_0_pulver_bedrifter_clicked();

    void on_ac_teknikk_clicked();
    void on_tilbake_1_pulver_ac_teknikk_clicked();

    void on_moss_jern_og_stanse_clicked();
    void on_tilbake_2_pulver_moss_jern_clicked();

    void on_rustfrie_berg_clicked();
    void on_tilbake_3_pulver_rustfrieBerg_clicked();

    void on_ivar_braathen_mekaniske_clicked();
    void on_tilbake_4_pulver_ivarBraathen_clicked();

private:
    Ui::pulverlakkering_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_pulver_bedrifter_clicked();
};

#endif // PULVERLAKKERING_SEND_H
