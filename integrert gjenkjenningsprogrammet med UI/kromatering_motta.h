#ifndef KROMATERING_MOTTA_H
#define KROMATERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class kromatering_motta;
}

class kromatering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit kromatering_motta(QWidget *parent = nullptr);
    ~kromatering_motta();

private slots:
    void on_tilbake_0_kromatering_bedrifter_clicked();

    void on_eloksal_clicked();
    void on_tilbake_1_eloksal_clicked();

    void on_gundersen_galvano_clicked();
    void on_tilbake_2_gundersen_clicked();

    void on_gundersen_galvano_vestby_clicked();
    void on_tilbake_3_gundersen_vestby_clicked();

    void on_eiker_overflateteknikk_clicked();
    void on_tilbake_4_eiker_clicked();

    void on_nct_clicked();
    void on_tilbake_5_nct_clicked();

private:
    Ui::kromatering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_kromatering_bedrifter_clicked();
};

#endif // KROMATERING_MOTTA_H
