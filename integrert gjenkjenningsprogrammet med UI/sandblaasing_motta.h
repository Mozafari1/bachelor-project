#ifndef SANDBLAASING_MOTTA_H
#define SANDBLAASING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class sandblaasing_motta;
}

class sandblaasing_motta : public QWidget
{
    Q_OBJECT

public:
    explicit sandblaasing_motta(QWidget *parent = nullptr);
    ~sandblaasing_motta();

private slots:
    void on_tilbake_0_sandblaasing_bedrifter_clicked();

    void on_bilskadesenteret_clicked();
    void on_tilbake_1_sandblaasing_bilskadesenteret_clicked();

    void on_nct_clicked();
    void on_tilbake_2_sandblaasing_nct_clicked();

private:
    Ui::sandblaasing_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_sandblaasing_bedrifter_clicked();
};

#endif // SANDBLAASING_MOTTA_H
