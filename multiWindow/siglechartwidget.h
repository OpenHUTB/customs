#ifndef SIGLECHARTWIDGET_H
#define SIGLECHARTWIDGET_H

#include <QWidget>
#include <QChartView>
#include <QLineSeries>
#include <QGridLayout>
#include <QChart>
#include <QComboBox>
#include <QVector>
#include <QPushButton>
#include <QFile>
#include <QTimer>
QT_CHARTS_USE_NAMESPACE

class SigleChartWidget : public QWidget
{
    Q_OBJECT
public:
    explicit SigleChartWidget(QWidget *parent = nullptr);
private:
    QGridLayout *baseLayout;
    QChartView *chartView;
    QChart *chart;
    QLineSeries* series;
    QPushButton *m_chooseFile;
    QTimer *timer;
    QVector<double> xlist;
    QVector<double> ylist;

    double minx,maxx;
    double miny,maxy;

    void initLayout();
    void vconnectSignals();
    QChart *createLineChart(QVector<double> xlist,QVector<double> ylist);

    std::pair<QVector<double>,QVector<double> > solveCsvFile(QFile &file);

private Q_SLOTS:
    void updateUI();
    void chooseCsvFile();
    void updateChart();
};

#endif // SIGLECHARTWIDGET_H
