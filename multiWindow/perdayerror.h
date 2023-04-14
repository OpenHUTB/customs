#ifndef PERDAYERROR_H
#define PERDAYERROR_H

#include <QWidget>
#include <QChartView>
#include <QLineSeries>
#include <QBarSeries>
#include <QGridLayout>
#include <QChart>
#include <QComboBox>
#include <QVector>
#include <QPushButton>
#include <QFile>
#include <string>
#include <QString>
#include <string>
using namespace std;
QT_CHARTS_USE_NAMESPACE
class PerDayError : public QWidget
{
    Q_OBJECT
public:
    explicit PerDayError(QWidget *parent = nullptr);
private:
    QGridLayout *baseLayout;
    QChartView* view;
    QChart* chart;
    double minx = -1;
    double miny = -1;
    double maxx = -1;
    double maxy = -1;
    QLineSeries *series1;
    QLineSeries *series2;
    QLineSeries *series3;

    std::pair<QVector<int>,QVector<double>> data_predict;
    std::pair<QVector<int>,QVector<double>> data_real;
    void initLayout();
    void vconnectSignals();
    void addLineChart(QVector<int> xlist,QVector<double> ylist,QString name,QLineSeries* line);
    void addAxis();
    std::pair<QVector<int>,QVector<double> > solveCsvFile(QFile &file);
    QVector<double> split(string s,char tag);
public slots:
    void updateUI();
    void process();
};

#endif // PERDAYERROR_H
