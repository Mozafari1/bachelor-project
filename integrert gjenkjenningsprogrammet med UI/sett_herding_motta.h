#ifndef SETT_HERDING_MOTTA_H
#define SETT_HERDING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class sett_herding_motta;
}

class sett_herding_motta : public QWidget
{
    Q_OBJECT

public:
    explicit sett_herding_motta(QWidget *parent = nullptr);
    ~sett_herding_motta();

private slots:
    void on_tilbake_0_sett_herding_bedrifter_clicked();

    void on_thune_produkter_raufoss_clicked();
    void on_tilbake_1_sett_herding_thune_clicked();

    void on_bodycote_clicked();
    void on_tilbake_2_sett_herding_bodycote_clicked();

private:
    Ui::sett_herding_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_sett_herding_bedrifter_clicked();
};

#endif // SETT_HERDING_MOTTA_H
