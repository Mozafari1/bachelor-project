#ifndef PLASMA_NITRERING_MOTTA_H
#define PLASMA_NITRERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class plasma_nitrering_motta;
}

class plasma_nitrering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit plasma_nitrering_motta(QWidget *parent = nullptr);
    ~plasma_nitrering_motta();

private slots:
    void on_tilbake_0_plasma_nitrering_bedrifter_clicked();

    void on_bodycote_clicked();
    void on_tilbake_1_plasma_nitrering_bodycote_clicked();

private:
    Ui::plasma_nitrering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_plasma_nitrering_bedrifter_clicked();
};

#endif // PLASMA_NITRERING_MOTTA_H
