#ifndef FORTINNING_MOTTA_H
#define FORTINNING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class fortinning_motta;
}

class fortinning_motta : public QWidget
{
    Q_OBJECT

public:
    explicit fortinning_motta(QWidget *parent = nullptr);
    ~fortinning_motta();

private slots:
    void on_tilbake_0_fortinning_bedrifter_clicked();

    void on_gundersen_clicked();
    void on_tilbake_1_fortinning_gundersen_clicked();

private:
    Ui::fortinning_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_fortinning_bedrifter_clicked();
};

#endif // FORTINNING_MOTTA_H
