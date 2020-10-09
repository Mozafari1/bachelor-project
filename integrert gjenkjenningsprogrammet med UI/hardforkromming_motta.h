#ifndef HARDFORKROMMING_MOTTA_H
#define HARDFORKROMMING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class hardforkromming_motta;
}

class hardforkromming_motta : public QWidget
{
    Q_OBJECT

public:
    explicit hardforkromming_motta(QWidget *parent = nullptr);
    ~hardforkromming_motta();

private slots:
    void on_tilbake_0_harforkromming_bedrifter_clicked();

    void on_norsk_everit_clicked();
    void on_tilbake_1_hardforkromming_norsk_everit_clicked();

    void on_eiker_clicked();
    void on_tilbake_2_hardforkromming_eiker_clicked();

private:
    Ui::hardforkromming_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_hardforkromming_bedrifter_clicked();
};

#endif // HARDFORKROMMING_MOTTA_H
