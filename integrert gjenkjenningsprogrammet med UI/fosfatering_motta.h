#ifndef FOSFATERING_MOTTA_H
#define FOSFATERING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class fosfatering_motta;
}

class fosfatering_motta : public QWidget
{
    Q_OBJECT

public:
    explicit fosfatering_motta(QWidget *parent = nullptr);
    ~fosfatering_motta();

private slots:
    void on_tilbake_0_fosfatering_bedrifter_clicked();

    void on_nct_clicked();
    void on_tilbake_1_fosfatering_nct_clicked();

    void on_gundersen_clicked();
    void on_tilbake_2_fosfatering_gundersen_clicked();

private:
    Ui::fosfatering_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_fosfatering_bedrifter_clicked();
};

#endif // FOSFATERING_MOTTA_H
