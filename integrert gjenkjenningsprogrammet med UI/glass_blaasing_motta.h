#ifndef GLASS_BLAASING_MOTTA_H
#define GLASS_BLAASING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class glass_blaasing_motta;
}

class glass_blaasing_motta : public QWidget
{
    Q_OBJECT

public:
    explicit glass_blaasing_motta(QWidget *parent = nullptr);
    ~glass_blaasing_motta();

private slots:
    void on_tilbake_0_glass_blaasing_bedrifter_clicked();

    void on_tronrud_clicked();
    void on_tilbake_1_glass_blaasing_tronrud_clicked();

private:
    Ui::glass_blaasing_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_glass_blaasing_bedrifter_clicked();
};

#endif // GLASS_BLAASING_MOTTA_H
