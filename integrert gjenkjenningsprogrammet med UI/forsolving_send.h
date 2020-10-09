#ifndef FORSOLVING_SEND_H
#define FORSOLVING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class forsolving_send;
}

class forsolving_send : public QWidget
{
    Q_OBJECT

public:
    explicit forsolving_send(QWidget *parent = nullptr);
    ~forsolving_send();

private slots:
    void on_tilbake_0_forsolving__bedrifter_clicked();

    void on_lie_clicked();
    void on_tilbake_1_forsolving_lie_clicked();

private:
    Ui::forsolving_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_forsolving_bedrifter_clicked();
};

#endif // FORSOLVING_SEND_H
