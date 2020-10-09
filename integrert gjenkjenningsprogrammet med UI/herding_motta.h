#ifndef HERDING_MOTTA_H
#define HERDING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class herding_motta;
}

class herding_motta : public QWidget
{
    Q_OBJECT

public:
    explicit herding_motta(QWidget *parent = nullptr);
    ~herding_motta();

private slots:
    void on_tilbake_0_herding_bedrifter_clicked();

    void on_thune_produkter_raufoss_clicked();
    void on_tilbake_1_herding_thune_clicked();

    void on_bodycote_clicked();
    void on_tilbake_2_herding_bodycote_clicked();

private:
    Ui::herding_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_herding_bedrifter_clicked();
};

#endif // HERDING_MOTTA_H
