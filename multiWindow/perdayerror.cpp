#include "perdayerror.h"
#include <QChartView>
#include <QLineSeries>
#include <QtMath>
#include <QValueAxis>
#include <QVector>
#include <QVBoxLayout>
#include <QPieSeries>
#include <QLabel>
#include <QDebug>
#include <QFileDialog>
#include <QFile>
#include <iostream>
#include <string>
#include <QBarSet>
#include <QString>
QT_CHARTS_USE_NAMESPACE

PerDayError::PerDayError(QWidget *parent) : QWidget(parent)
{
    initLayout();
    vconnectSignals();

    process();
    addAxis();
}

void PerDayError::initLayout()
{
    QGridLayout *baseLayout = new QGridLayout();
    chart = new QChart();
    view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);//设置绘图的时候抗锯齿
    series1 = new QLineSeries;
    series2 = new QLineSeries;
    series3 = new QLineSeries;
    baseLayout->addWidget(view,0,0);
    this->setLayout(baseLayout);
}

void PerDayError::vconnectSignals()
{

}

void PerDayError::addLineChart(QVector<int> xlist, QVector<double> ylist, QString name,QLineSeries* line)
{
    chart->setTitle("每日用电量真实、预测、误差对比图");
    //非法检查
    if(xlist.size() != ylist.size() || xlist.size() == 0) return ;
    qDebug()<<"ThemeWidget::createLineChart: 已经通过了非法检查\n";
    line->setName(name);
    chart->addSeries(line);
    int n = xlist.size();
    if(minx < 0 || miny < 0){
        minx = 1,maxx = 365;
        miny = ylist[0],maxy = ylist[0];
    }
    for(int i = 0;i<n;i++){
        line->append(xlist[i],ylist[i]);
        miny = std::min(miny,ylist[i]);
        maxy = std::max(maxy,ylist[i]);
    }
    //创建坐标轴
//    qDebug()<<minx<<" "<<maxx<<" "<<miny<<" "<<maxy<<'\n';
}

void PerDayError::addAxis()
{
    QValueAxis *axisX = new QValueAxis;
    axisX->setTitleText("天");
    axisX->setRange(0,365);
    axisX->setTickCount(20);
    axisX->setLabelFormat("%d");
    chart->setAxisX(axisX,series1);
    chart->setAxisX(axisX,series2);
    chart->setAxisX(axisX,series3);

    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("用电量");
    axisY->setRange(miny,maxy);
    axisY->setTickCount(10);
    chart->setAxisY(axisY,series1);
    chart->setAxisY(axisY,series2);
    chart->setAxisY(axisY,series3);
}

std::pair<QVector<int>, QVector<double> > PerDayError::solveCsvFile(QFile &file)
{
    qDebug()<<"进入到了PerDayError::solveCsvFile函数\n";
    std::string s;
    QVector<int> xlist;
    QVector<double> ylist;
    QTextStream in(&file);
    int dayid = 0;
    while (!in.atEnd()) {
        // 处理每一行数据
        QString line = in.readLine();
        std::string  s = line.toStdString();
        QVector<double> hours = split(s,',');
        double daysum = 0;
        for(int i = 0;i<24;i++) daysum += hours[i];
        ++dayid;
        xlist.append(dayid);
        ylist.append(daysum);
    }
    std::pair<QVector<int>,QVector<double>> ans = {xlist,ylist};
    qDebug()<<"退出了MonthPredictChartWidget::solveCsvFile函数\n";
    return ans;
}

QVector<double> PerDayError::split(string s, char tag)
{
    QVector<double> ans;
    string cur;
    for(char c:s){
        if(c == tag){
            ans.append(std::stod(cur));
            cur = "";
        }else{
            cur += c;
        }
    }
    ans.append(std::stod(cur));
    return ans;
}

void PerDayError::updateUI()
{

}

void PerDayError::process()
{
    qDebug()<<"进入到了MonthPredictChartWidget::chooseCsvFile()函数----------\n";
    QFile file1;
    file1.setFileName("E:/QT/charts/multiWindow/data/2002label.csv");
    if (!file1.open(QIODevice::ReadOnly | QIODevice::Text)){
        qDebug()<<"打开失败!\n";
        return; // 打开失败
    }
    std::pair<QVector<int>,QVector<double>> pos1 =  solveCsvFile(file1);

    QFile file2;
    file2.setFileName("E:/QT/charts/multiWindow/data/2002prediction.csv");
    if (!file2.open(QIODevice::ReadOnly | QIODevice::Text)){
        qDebug()<<"打开失败!\n";
        return; // 打开失败
    }
    std::pair<QVector<int>,QVector<double>> pos2 =  solveCsvFile(file2);
    QVector<double> difference;
    for(int i = 0;i<pos1.first.size();i++){
        difference.append(abs(pos1.second.at(i) - pos2.second.at(i)));
    }
    addLineChart(pos1.first,pos1.second,"真实值",series1);
    addLineChart(pos2.first,pos2.second,"预测值",series2);
    addLineChart(pos1.first,difference,"用电量误差",series3);
}


