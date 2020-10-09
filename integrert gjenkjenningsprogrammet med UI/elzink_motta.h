#ifndef ELZINK_MOTTA_H
#define ELZINK_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class elzink_motta;
}

class elzink_motta : public QWidget
{
    Q_OBJECT

public:
    explicit elzink_motta(QWidget *parent = nullptr);
    ~elzink_motta();

private slots:
    void on_gundersen_galvano_vestby_clicked();

    void on_eiker_overflateteknikk_clicked();

    void on_nct_clicked();

    void on_tilbake_1_elzink_gundersen_vestby_clicked();

    void on_tilbake_2_elzink_eiker_clicked();

    void on_tilbake_3_elzink_honefoss_clicked();


    void on_tilbake_0_elzink_bedrifter_clicked();

private:
    Ui::elzink_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_elzink_bedrifter_clicked();
};

#endif // ELZINK_MOTTA_H
