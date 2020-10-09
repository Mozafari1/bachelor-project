#ifndef ELOKSERING_MOTTA_H
#define ELOKSERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class eloksering_motta;
}

class eloksering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit eloksering_motta(QWidget *parent = nullptr);
    ~eloksering_motta();

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
    Ui::eloksering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;

signals:
    void tilbake_0_eloksering_bedrifter_clicked();
};

#endif // ELOKSERING_MOTTA_H
