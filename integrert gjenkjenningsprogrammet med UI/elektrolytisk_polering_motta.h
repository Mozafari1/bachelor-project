#ifndef ELEKTROLYTISK_POLERING_MOTTA_H
#define ELEKTROLYTISK_POLERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class elektrolytisk_polering_motta;
}

class elektrolytisk_polering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit elektrolytisk_polering_motta(QWidget *parent = nullptr);
    ~elektrolytisk_polering_motta();

private slots:
    void on_tilbake_0_elektrolytisk_polering__bedrifter_clicked();

    void on_gundersen_vestby_clicked();
    void on_tilbake_1_elektrolytisk_polering_gundersen_clicked();

private:
    Ui::elektrolytisk_polering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;

signals:
    void tilbake_0_elektrolytisk_polering_bedrifter_clicked();
};

#endif // ELEKTROLYTISK_POLERING_MOTTA_H
