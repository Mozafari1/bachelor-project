#ifndef BRUNERING_MOTTA_H
#define BRUNERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class brunering_motta;
}

class brunering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit brunering_motta(QWidget *parent = nullptr);
    ~brunering_motta();

private slots:
    void on_tilbake_0_brunering_bedrifter_clicked();

    void on_borsemakern_clicked();
    void on_tilbake_1_brunering_borsemakern_clicked();

    void on_eiker_clicked();
    void on_tilbake_2_brunering_eiker_clicked();

private:
    Ui::brunering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;

signals:
    void tilbake_0_brunering_bedrifter_clicked();
};

#endif // BRUNERING_MOTTA_H
