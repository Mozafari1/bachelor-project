#ifndef GULLBELEGGING_SEND_H
#define GULLBELEGGING_SEND_H

#include <QWidget>
#include <gjenkjenning_send.h>
namespace Ui {
class gullbelegging_send;
}

class gullbelegging_send : public QWidget
{
    Q_OBJECT

public:
    explicit gullbelegging_send(QWidget *parent = nullptr);
    ~gullbelegging_send();

private slots:
    void on_tilbake_0_gullbelegging__bedrifter_clicked();

    void on_jens_clicked();
    void on_tilbake_1_gullbelegging_jens_clicked();

private:
    Ui::gullbelegging_send *ui;
    class gjenkjenning_send _GjenkjennSend;
signals:
    void tilbake_0_gullbelegging_bedrifter_clicked();
};

#endif // GULLBELEGGING_SEND_H
