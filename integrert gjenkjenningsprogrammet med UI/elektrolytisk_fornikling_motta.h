#ifndef ELEKTROLYTISK_FORNIKLING_MOTTA_H
#define ELEKTROLYTISK_FORNIKLING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class elektrolytisk_fornikling_motta;
}

class elektrolytisk_fornikling_motta : public QWidget
{
    Q_OBJECT

public:
    explicit elektrolytisk_fornikling_motta(QWidget *parent = nullptr);
    ~elektrolytisk_fornikling_motta();

private slots:
    void on_tilbake_0_elektrolytisk_fornikling_bedrifter_clicked();

    void on_honefoss_clicked();
    void on_tilbake_1_elektrolytisk_fornikling_honefoss_clicked();

    void on_gundersen_galvano_clicked();
    void on_tilbake_2_elektrolytisk_fornikling_gundersen_clicked();

    void on_norsk_everit_clicked();
    void on_tilbake_3_elektrolytisk_fornikling_norsk_everit_clicked();

private:
    Ui::elektrolytisk_fornikling_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;

signals:
    void tilbake_0_elektrolytisk_fornikling_bedrifter_clicked();
};

#endif // ELEKTROLYTISK_FORNIKLING_MOTTA_H
