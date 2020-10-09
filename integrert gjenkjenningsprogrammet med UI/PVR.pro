QT       += core gui sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    brunering_motta.cpp \
    brunering_send.cpp \
    elektrolytisk_fornikling_motta.cpp \
    elektrolytisk_fornikling_send.cpp \
    elektrolytisk_polering_motta.cpp \
    elektrolytisk_polering_send.cpp \
    eloksering_motta.cpp \
    eloksering_send.cpp \
    elzink_motta.cpp \
    elzink_send.cpp \
    forsolving_motta.cpp \
    forsolving_send.cpp \
    fortinning_motta.cpp \
    fortinning_send.cpp \
    fosfatering_motta.cpp \
    fosfatering_send.cpp \
    galvanisering_motta.cpp \
    galvanisering_send.cpp \
    gjenkjenning_motta.cpp \
    gjenkjenning_send.cpp \
    glass_blaasing_motta.cpp \
    glass_blaasing_send.cpp \
    gullbelegging_motta.cpp \
    gullbelegging_send.cpp \
    hardforkromming_motta.cpp \
    hardforkromming_send.cpp \
    herding_motta.cpp \
    herding_send.cpp \
    kromatering_motta.cpp \
    kromatering_send.cpp \
    main.cpp \
    mainwindow.cpp \
    mottar.cpp \
    nitrering_motta.cpp \
    nitrering_send.cpp \
    offshore_lakkering_motta.cpp \
    offshore_lakkering_send.cpp \
    plasma_nitrering_motta.cpp \
    plasma_nitrering_send.cpp \
    polering_motta.cpp \
    polering_send.cpp \
    pulverlakkering_motta.cpp \
    pulverlakkering_send.cpp \
    sandblaasing_motta.cpp \
    sandblaasing_send.cpp \
    sender.cpp \
    sett_herding_motta.cpp \
    sett_herding_send.cpp \
    sliping_motta.cpp \
    sliping_send.cpp \
    syrevask_motta.cpp \
    syrevask_send.cpp \
    traadskjaering_motta.cpp \
    traadskjaering_send.cpp \
    vaatlakkering_motta.cpp \
    vaatlakkering_send.cpp

HEADERS += \
    brunering_motta.h \
    brunering_send.h \
    elektrolytisk_fornikling_motta.h \
    elektrolytisk_fornikling_send.h \
    elektrolytisk_polering_motta.h \
    elektrolytisk_polering_send.h \
    eloksering_motta.h \
    eloksering_send.h \
    elzink_motta.h \
    elzink_send.h \
    forsolving_motta.h \
    forsolving_send.h \
    fortinning_motta.h \
    fortinning_send.h \
    fosfatering_motta.h \
    fosfatering_send.h \
    galvanisering_motta.h \
    galvanisering_send.h \
    gjenkjenning_motta.h \
    gjenkjenning_send.h \
    glass_blaasing_motta.h \
    glass_blaasing_send.h \
    gullbelegging_motta.h \
    gullbelegging_send.h \
    hardforkromming_motta.h \
    hardforkromming_send.h \
    herding_motta.h \
    herding_send.h \
    kromatering_motta.h \
    kromatering_send.h \
    mainwindow.h \
    mottar.h \
    nitrering_motta.h \
    nitrering_send.h \
    offshore_lakkering_motta.h \
    offshore_lakkering_send.h \
    plasma_nitrering_motta.h \
    plasma_nitrering_send.h \
    polering_motta.h \
    polering_send.h \
    pulverlakkering_motta.h \
    pulverlakkering_send.h \
    sandblaasing_motta.h \
    sandblaasing_send.h \
    sender.h \
    sett_herding_motta.h \
    sett_herding_send.h \
    sliping_motta.h \
    sliping_send.h \
    syrevask_motta.h \
    syrevask_send.h \
    traadskjaering_motta.h \
    traadskjaering_send.h \
    vaatlakkering_motta.h \
    vaatlakkering_send.h

FORMS += \
    brunering_motta.ui \
    brunering_send.ui \
    elektrolytisk_fornikling_motta.ui \
    elektrolytisk_fornikling_send.ui \
    elektrolytisk_polering_motta.ui \
    elektrolytisk_polering_send.ui \
    eloksering_motta.ui \
    eloksering_send.ui \
    elzink_motta.ui \
    elzink_send.ui \
    forsolving_motta.ui \
    forsolving_send.ui \
    fortinning_motta.ui \
    fortinning_send.ui \
    fosfatering_motta.ui \
    fosfatering_send.ui \
    galvanisering_motta.ui \
    galvanisering_send.ui \
    gjenkjenning_motta.ui \
    gjenkjenning_send.ui \
    glass_blaasing_motta.ui \
    glass_blaasing_send.ui \
    gullbelegging_motta.ui \
    gullbelegging_send.ui \
    hardforkromming_motta.ui \
    hardforkromming_send.ui \
    herding_motta.ui \
    herding_send.ui \
    kromatering_motta.ui \
    kromatering_send.ui \
    kuleblaasing_motta.ui \
    kuleblaasing_send.ui \
    mainwindow.ui \
    mottar.ui \
    nitrering_motta.ui \
    nitrering_send.ui \
    offshore_lakkering_motta.ui \
    offshore_lakkering_send.ui \
    plasma_nitrering_motta.ui \
    plasma_nitrering_send.ui \
    polering_motta.ui \
    polering_send.ui \
    pulverlakkering_motta.ui \
    pulverlakkering_send.ui \
    sandblaasing_motta.ui \
    sandblaasing_send.ui \
    sender.ui \
    sett_herding_motta.ui \
    sett_herding_send.ui \
    sliping_motta.ui \
    sliping_send.ui \
    syrevask_motta.ui \
    syrevask_send.ui \
    traadskjaering_motta.ui \
    traadskjaering_send.ui \
    vaatlakkering_motta.ui \
    vaatlakkering_send.ui

    LIBS += -LC:\OpenCV_DIR\build_files\install\x64\vc16\lib -lopencv_core3410d -lopencv_imgproc3410d -lopencv_highgui3410d -lopencv_imgcodecs3410d -lopencv_videoio3410d -lopencv_video3410d -lopencv_calib3d3410d -lopencv_photo3410d -lopencv_features2d3410d -lopencv_xfeatures2d3410d

    INCLUDEPATH += C:\OpenCV_DIR\build_files\install\include

    DEPENDPATH += C:\OpenCV_DIR\build_files\install\include

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


CONFIG += enable_encoder_qr_code
include(C:\OpenCV_DIR\src\QZXing.pri)

win32: LIBS += -L$$PWD/../../../Users/m_rah/Desktop/lib/ -llibmysql

INCLUDEPATH += $$PWD/../../../Users/m_rah/Desktop
DEPENDPATH += $$PWD/../../../Users/m_rah/Desktop
