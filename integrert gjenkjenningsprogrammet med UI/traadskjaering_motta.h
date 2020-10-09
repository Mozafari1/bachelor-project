#ifndef TRAADSKJAERING_MOTTA_H
#define TRAADSKJAERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class traadskjaering_motta;
}

class traadskjaering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit traadskjaering_motta(QWidget *parent = nullptr);
    ~traadskjaering_motta();

private slots:
    void on_tilbake_0_traadskjaering_bedrifter_clicked();

    void on_sandsje_clicked();
    void on_tilbake_1_traadskjaering_sandsje_clicked();

    void on_mvp_clicked();
    void on_tilbake_2_traadskjaering_mvp_clicked();

private:
    Ui::traadskjaering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_traadskjaering_bedrifter_clicked();
};

#endif // TRAADSKJAERING_MOTTA_H
