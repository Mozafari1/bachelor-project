#ifndef SYREVASK_MOTTA_H
#define SYREVASK_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class syrevask_motta;
}

class syrevask_motta : public QWidget
{
    Q_OBJECT

public:
    explicit syrevask_motta(QWidget *parent = nullptr);
    ~syrevask_motta();

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
    Ui::syrevask_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_syrevask_bedrifter_clicked();
};

#endif // SYREVASK_MOTTA_H
