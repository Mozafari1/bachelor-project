#ifndef GULLBELEGGING_MOTTA_H
#define GULLBELEGGING_MOTTA_H

#include <QWidget>
#include<gjenkjenning_motta.h>
namespace Ui {
class gullbelegging_motta;
}

class gullbelegging_motta : public QWidget
{
    Q_OBJECT

public:
    explicit gullbelegging_motta(QWidget *parent = nullptr);
    ~gullbelegging_motta();

private slots:
    void on_tilbake_0_gullbelegging__bedrifter_clicked();

    void on_jens_clicked();
    void on_tilbake_1_gullbelegging_jens_clicked();

private:
    Ui::gullbelegging_motta *ui;
    class gjenkjenning_motta _GjenkjennMotta;
signals:
    void tilbake_0_gullbelegging_bedrifter_clicked();
};

#endif // GULLBELEGGING_MOTTA_H
