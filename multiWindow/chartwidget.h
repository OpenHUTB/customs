#ifndef CHARTWIDGET_H
#define CHARTWIDGET_H

#include <QChartView>
#include <QLineSeries>
#include <QGridLayout>
#include <QChart>
#include <QComboBox>
#include <QVector>
#include <QPushButton>
#include <QFile>
QT_CHARTS_USE_NAMESPACE

class ChartWidget : public QWidget
{
    Q_OBJECT
public:
    ChartWidget(QWidget *parent = 0);
    ~ChartWidget();

private:
    QGridLayout *baseLayout;
    QChartView *chartView1;
    QChartView *chartView2;
    QComboBox *m_themeComboBox;
    QChart *chart1;
    QChart *chart2;
    QPushButton *m_chooseFile;

    void initLayout();
    void vconnectSignals();
    QChart *createLineChart(QVector<double> xlist,QVector<double> ylist);
    QChart *createPieChart() const;
    QComboBox *createThemeBox()const;


    std::pair<QVector<double>,QVector<double> > solveCsvFile(QFile &file);


private Q_SLOTS:
    void updateUI();
    void chooseCsvFile();
};

#endif // CHARTWIDGET_H
