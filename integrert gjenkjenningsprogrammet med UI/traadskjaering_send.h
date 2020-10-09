#ifndef TRAADSKJAERING_SEND_H
#define TRAADSKJAERING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class traadskjaering_send;
}

class traadskjaering_send : public QWidget
{
    Q_OBJECT

public:
    explicit traadskjaering_send(QWidget *parent = nullptr);
    ~traadskjaering_send();

private slots:
    void on_tilbake_0_traadskjaering_bedrifter_clicked();

    void on_sandsje_clicked();
    void on_tilbake_1_traadskjaering_sandsje_clicked();

    void on_mvp_clicked();
    void on_tilbake_2_traadskjaering_mvp_clicked();

private:
    Ui::traadskjaering_send *ui;
    class gjenkjenning_send _GjenkjennSend;

signals:
    void tilbake_0_traadskjaering_bedrifter_clicked();
};

#endif // TRAADSKJAERING_SEND_H
