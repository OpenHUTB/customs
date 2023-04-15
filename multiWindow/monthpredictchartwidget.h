#ifndef MONTHPREDICTCHARTWIDGET_H
#define MONTHPREDICTCHARTWIDGET_H

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
using namespace std;
QT_CHARTS_USE_NAMESPACE

class MonthPredictChartWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MonthPredictChartWidget(QWidget *parent = nullptr);

private:
    QGridLayout *baseLayout;
    QChartView* view;
    QChart* chart;
    QBarSeries* series;
    QPushButton *m_chooseFile;
    int series_cnt = 0;
    int highest = 0;

    std::pair<QVector<int>,QVector<double>> data_predict;
    std::pair<QVector<int>,QVector<double>> data_real;

    void initLayout();
    void vconnectSignals();
    void addBarChart(QVector<int> xlist,QVector<double> ylist,QString name);
    void addAxis();
    std::pair<QVector<int>,QVector<double> > solveCsvFile(QFile &file);
    QVector<double> split(string s,char tag);
private Q_SLOTS:
    void updateUI();
    void process();
};

#endif // MONTHPREDICTCHARTWIDGET_H
