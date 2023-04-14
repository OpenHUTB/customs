#include "monthpredictchartwidget.h"
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

MonthPredictChartWidget::MonthPredictChartWidget(QWidget *parent) : QWidget(parent)
{
    initLayout();
    vconnectSignals();

    process();
}

void MonthPredictChartWidget::initLayout()
{
    QGridLayout *baseLayout = new QGridLayout();
//    QHBoxLayout *settingsLayout = new QHBoxLayout();
    chart = new QChart();
    view = new QChartView(chart);
    series = new QBarSeries();
//    m_chooseFile = new QPushButton("选择预测结果");
//    settingsLayout->addWidget(m_chooseFile);
//    baseLayout->addLayout(settingsLayout,0,0,1,1);
    baseLayout->addWidget(view,1,0);
    this->setLayout(baseLayout);
}

void MonthPredictChartWidget::vconnectSignals()
{
    //connect(m_chooseFile,SIGNAL(clicked()),this,SLOT(chooseCsvFile()));
}

void MonthPredictChartWidget::addBarChart(QVector<int> xlist, QVector<double> ylist,QString name)
{
    QBarSet* set = new QBarSet(name);
    for(int i = 0;i<ylist.size();i++){
        *set<<ylist[i];
    }
    series->append(set);
    chart->addSeries(series);
}

void MonthPredictChartWidget::addAxis()
{
   QValueAxis *axisX = new QValueAxis();
   axisX->setRange(0, 12);
   axisX->setTickCount(13);
   axisX->setLabelFormat("%d");
   axisX->setTitleText("月份");
   chart->setAxisX(axisX,series);

   QValueAxis *axisY = new QValueAxis();
   axisY->setRange(0, highest);
   axisY->setLabelFormat("%d");
   axisY->setTitleText("用电量");
   axisY->setTickCount(20);
   chart->setAxisY(axisY,series);
}

std::pair<QVector<int>, QVector<double> > MonthPredictChartWidget::solveCsvFile(QFile &file)
{
    qDebug()<<"进入到了MonthPredictChartWidget::solveCsvFile函数\n";
    std::string s;
    QVector<int> xlist;
    QVector<double> ylist;
    QTextStream in(&file);
    int monCnt[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    int id = 0,cur = 0,monSum = 0;
    while (!in.atEnd()) {
        // 处理每一行数据
        QString line = in.readLine();
        std::string  s = line.toStdString();
        QVector<double> hours = split(s,',');
        double daysum = 0;
        for(int i = 0;i<24;i++) daysum += hours[i];
        monSum += daysum;
        if(++cur > monCnt[id]){
            cur = 0;
            ++id;
            xlist.append(id);
            ylist.append(monSum);
            highest = max(highest,monSum);
            monSum = 0;
        }
    }
    std::pair<QVector<int>,QVector<double>> ans = {xlist,ylist};
    qDebug()<<"退出了MonthPredictChartWidget::solveCsvFile函数\n";
    return ans;
}

QVector<double> MonthPredictChartWidget::split(string s,char tag)
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

void MonthPredictChartWidget::updateUI()
{

}

void MonthPredictChartWidget::process()
{
    qDebug()<<"进入到了MonthPredictChartWidget::chooseCsvFile()函数----------\n";
//    QString fileName = QFileDialog::getOpenFileName(this,QStringLiteral("选择文件"),"F:",QStringLiteral("excel文件(*csv)"));
//    qDebug()<<"读取的文件是："<<fileName<<'\n';
//    if(fileName.size() == 0){
//        qDebug()<<"取消了选择文件\n";
//        return ;
//    }
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
    addAxis();
    addBarChart(pos1.first,pos1.second,"真实值");
    addBarChart(pos2.first,pos2.second,"预测值");
}

