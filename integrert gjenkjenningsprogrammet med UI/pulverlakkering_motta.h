#ifndef PULVERLAKKERING_MOTTA_H
#define PULVERLAKKERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class pulverlakkering_motta;
}

class pulverlakkering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit pulverlakkering_motta(QWidget *parent = nullptr);
    ~pulverlakkering_motta();

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
    Ui::pulverlakkering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_pulver_bedrifter_clicked();
};

#endif // PULVERLAKKERING_MOTTA_H
