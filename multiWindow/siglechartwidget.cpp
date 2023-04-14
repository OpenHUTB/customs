#include "siglechartwidget.h"
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
#include <unistd.h>
QT_CHARTS_USE_NAMESPACE

SigleChartWidget::SigleChartWidget(QWidget *parent) : QWidget(parent)
{
    initLayout();
    vconnectSignals();

}

void SigleChartWidget::initLayout()
{
    QGridLayout *baseLayout = new QGridLayout();
    QHBoxLayout *settingsLayout = new QHBoxLayout();
    timer = new QTimer(this);
    m_chooseFile = new QPushButton("选择训练数据");
    settingsLayout->addWidget(m_chooseFile);
    //初始化图表
    chart = new QChart();
    chart->setTitle("Line Chart");
    series = new QLineSeries;
    series->setName("模型误差值");
    chart->addSeries(series);
    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);//设置绘图的时候抗锯齿
    baseLayout->addLayout(settingsLayout,0,0,1,1);
    baseLayout->addWidget(chartView,1,0);
    this->setLayout(baseLayout);
}

void SigleChartWidget::vconnectSignals()
{
    connect(m_chooseFile,SIGNAL(clicked()),this,SLOT(chooseCsvFile()));
    connect(timer, SIGNAL(timeout()), this, SLOT(updateChart()));
}

QChart *SigleChartWidget::createLineChart(QVector<double> xlist, QVector<double> ylist)
{
    //非法检查
    if(xlist.size() != ylist.size() || xlist.size() == 0) return chart;
//    qDebug()<<"ThemeWidget::createLineChart: 已经通过了非法检查\n";
    int n = xlist.size();
    if(series->count() == 0){
        minx = xlist[0],maxx = xlist[0];
        miny = ylist[0],maxy = ylist[0];
    }
    for(int i = 0;i<n;i++){
        series->append(xlist[i],ylist[i]);
        minx = std::min(minx,xlist[i]);
        maxx = std::max(maxx,xlist[i]);
        miny = std::min(miny,ylist[i]);
        maxy = std::max(maxy,ylist[i]);
    }
    //创建坐标轴
//    qDebug()<<minx<<" "<<maxx<<" "<<miny<<" "<<maxy<<'\n';
    QValueAxis *axisX = new QValueAxis;
    axisX->setRange(minx,maxx);
    chart->setAxisX(axisX,series);

    QValueAxis *axisY = new QValueAxis;
    axisY->setRange(miny,maxy);
    chart->setAxisY(axisY,series);
    return chart;
}

std::pair<QVector<double>, QVector<double> > SigleChartWidget::solveCsvFile(QFile &file)
{
    qDebug()<<"进入到了ThemeWidget::solveCsvFile函数\n";
//    freopen(filename.toStdString().c_str(),"r",stdin);
    std::string s;
    QVector<double> xlist,ylist;
    QTextStream in(&file);
    in.readLine();
    while (!in.atEnd()) {
        // 处理每一行数据
        QString line = in.readLine();
        std::string  s = line.toStdString();
        int pos = s.find(",");
        double x = std::stod(s.substr(0,pos));
        double y = std::stod(s.substr(pos+1));
//        qDebug()<<x<<" "<<y<<'\n';
        xlist.append(x);
        ylist.append(y);
    }
    std::pair<QVector<double>,QVector<double>> ans = {xlist,ylist};
    return ans;
}

void SigleChartWidget::updateUI()
{

}

void SigleChartWidget::chooseCsvFile()
{
    qDebug()<<"进入到了ThemeWidget::chooseCsvFile函数----------\n";
    QStringList fileNames = QFileDialog::getOpenFileNames(this, tr("Select Files"), "D:/data/", tr("All Files (*.*)"));
    if (!fileNames.isEmpty()) {
        // 处理选择的文件
        foreach(QString fileName, fileNames) {
            // do something with fileName
        }
    }else return ;
    QFile file;
    file.setFileName("E:/QT/charts/multiWindow/data/train_loss.csv");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return; // 打开失败
    std::pair<QVector<double>,QVector<double>> pos =  solveCsvFile(file);
    xlist = pos.first;
    ylist = pos.second;
    timer->start(200);
//    if(chartView1 != nullptr) delete chartView1;


//    QLineSeries *series =  (QLineSeries *)chartView1->chart()->series().at(0);
//    QVector<QPointF> pts = series->pointsVector();
//    for(int i = 0;i<pts.size();i++){
//        double x = pts[i].x();
//        double y = pts[i].y();
//        qDebug()<<x<<" "<<y<<'\n';
    //    }
}

void SigleChartWidget::updateChart()
{
    qDebug()<<"进入到了SigleChartWidget::updateChart()函数\n";
    int count = 10;
    int start = series->count();
    QVector<double> curx,cury;
    for(int i = start;i <xlist.size() && i < start + count;i++){
        curx.append(xlist[i]);
        cury.append(ylist[i]);
    }
    createLineChart(curx,cury);
    if(series->count() == xlist.size()){
        qDebug()<<"count = "<<series->count()<<" xlist.size() = "<<xlist.size()<<'\n';
        timer->stop();
    }
}
