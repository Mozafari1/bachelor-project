#ifndef FORTINNING_SEND_H
#define FORTINNING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class fortinning_send;
}

class fortinning_send : public QWidget
{
    Q_OBJECT

public:
    explicit fortinning_send(QWidget *parent = nullptr);
    ~fortinning_send();

private slots:
    void on_tilbake_0_fortinning_bedrifter_clicked();

    void on_gundersen_clicked();
    void on_tilbake_1_fortinning_gundersen_clicked();

private:
    Ui::fortinning_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_fortinning_bedrifter_clicked();
};

#endif // FORTINNING_SEND_H
