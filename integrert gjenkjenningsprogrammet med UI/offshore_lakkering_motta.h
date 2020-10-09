#ifndef OFFSHORE_LAKKERING_MOTTA_H
#define OFFSHORE_LAKKERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class offshore_lakkering_motta;
}

class offshore_lakkering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit offshore_lakkering_motta(QWidget *parent = nullptr);
    ~offshore_lakkering_motta();

private slots:
    void on_nct_clicked();

    void on_tilbake_0_offshore_nct_clicked();
    void on_tilbake_0_offshore_bedrifter_clicked();

private:
    Ui::offshore_lakkering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_offshore_bedrifter_clicked();
};

#endif // OFFSHORE_LAKKERING_MOTTA_H
