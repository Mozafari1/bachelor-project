#ifndef NITRERING_MOTTA_H
#define NITRERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class nitrering_motta;
}

class nitrering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit nitrering_motta(QWidget *parent = nullptr);
    ~nitrering_motta();

private slots:
    void on_tilbake_0_nitrering_bedrifter_clicked();

    void on_thune_produkter_raufoss_clicked();
    void on_tilbake_1_nitrering_thune_clicked();

    void on_bodycote_clicked();
    void on_tilbake_2_nitrering_bodycote_clicked();

private:
    Ui::nitrering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_nitrering_bedrifter_clicked();
};

#endif // NITRERING_MOTTA_H
