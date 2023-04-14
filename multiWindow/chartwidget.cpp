#include "chartwidget.h"
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
QT_CHARTS_USE_NAMESPACE

ChartWidget::ChartWidget(QWidget *parent) : QWidget(parent),m_themeComboBox(createThemeBox())
{
    initLayout();
    vconnectSignals();
}

ChartWidget::~ChartWidget()
{

}

void ChartWidget::initLayout()
{
    QGridLayout *baseLayout = new QGridLayout();
    QHBoxLayout *settingsLayout = new QHBoxLayout();
    m_chooseFile = new QPushButton("选择csv文件");

    settingsLayout->addWidget(new QLabel("主题： "));
    settingsLayout->addWidget(m_themeComboBox);
    settingsLayout->addWidget(m_chooseFile);

    baseLayout->addLayout(settingsLayout,0,0,1,1);
    QVector<double> v1 = {1,2,3},v2 = {2,3,4};
    chartView1 = new QChartView(createLineChart(v1,v2));
    baseLayout->addWidget(chartView1,1,0);

    chartView2 = new QChartView(createPieChart());
    baseLayout->addWidget(chartView2,1,1);
    qDebug()<<this->layout();
    this->setLayout(baseLayout);
}

void ChartWidget::vconnectSignals()
{
    connect(m_themeComboBox,SIGNAL(currentIndexChanged(int)),this,SLOT(updateUI()));
    connect(m_chooseFile,SIGNAL(clicked()),this,SLOT(chooseCsvFile()));
}

QChart *ChartWidget::createLineChart(QVector<double> xlist, QVector<double> ylist)
{
    QChart *chart = new QChart();
    chart->setTitle("Line Chart");
    //非法检查
    if(xlist.size() != ylist.size() || xlist.size() == 0) return chart;
    qDebug()<<"ThemeWidget::createLineChart: 已经通过了非法检查\n";
    QLineSeries *series = new QLineSeries;
    series->setName("csv文件数据");
    chart->addSeries(series);

    int n = xlist.size();
    double minx = xlist[0],maxx = xlist[0];
    double miny = ylist[0],maxy = ylist[0];
    for(int i = 0;i<n;i++){
        series->append(xlist[i],ylist[i]);
        minx = std::min(minx,xlist[i]);
        maxx = std::max(maxx,xlist[i]);
        miny = std::min(miny,ylist[i]);
        maxy = std::max(maxy,ylist[i]);
    }
    //创建坐标轴
    qDebug()<<minx<<" "<<maxx<<" "<<miny<<" "<<maxy<<'\n';
    QValueAxis *axisX = new QValueAxis;
    axisX->setRange(minx,maxx);
    chart->setAxisX(axisX,series);

    QValueAxis *axisY = new QValueAxis;
    axisY->setRange(miny,maxy);
    chart->setAxisY(axisY,series);
    return chart;
}

QChart *ChartWidget::createPieChart() const
{
    QChart *chart = new QChart();
    chart->setTitle("Pie chart");
    QPieSeries *series = new QPieSeries();
    chart->addSeries(series);
    series->append("Chrome", 20);
    series->append("Firefox", 15);
    series->append("IE", 10);
    series->append("Safari", 5);

    foreach(QPieSlice *slice, series->slices()){
        QString label = QString("%1\n%2%").arg(slice->label()).
                arg(100 * slice->percentage(),0,'f',1);

        slice->setLabel(label);
    }
    return chart;
}

QComboBox *ChartWidget::createThemeBox() const
{
    QComboBox *themeComboBox = new QComboBox();
    themeComboBox->addItem("Light",QChart::ChartThemeLight);
    themeComboBox->addItem("Blue Cerulean",QChart::ChartThemeLight);
    themeComboBox->addItem("Dark", QChart::ChartThemeDark);
    themeComboBox->addItem("Brown Sand", QChart::ChartThemeBrownSand);
    themeComboBox->addItem("Blue NCS", QChart::ChartThemeBlueNcs);
    themeComboBox->addItem("High Contrast", QChart::ChartThemeHighContrast);
    themeComboBox->addItem("Blue Icy", QChart::ChartThemeBlueIcy);
    themeComboBox->addItem("Qt", QChart::ChartThemeQt);
    return themeComboBox;
}

std::pair<QVector<double>, QVector<double> > ChartWidget::solveCsvFile(QFile &file)
{
    qDebug()<<"进入到了ThemeWidget::solveCsvFile函数\n";
//    freopen(filename.toStdString().c_str(),"r",stdin);
    std::string s;
    QVector<double> xlist,ylist;
    QTextStream in(&file);
    while (!in.atEnd()) {
        // 处理每一行数据
        QString line = in.readLine();
        std::string  s = line.toStdString();
        int pos = s.find(",");
        double x = std::stod(s.substr(0,pos));
        double y = std::stod(s.substr(pos+1));
        qDebug()<<x<<" "<<y<<'\n';
        xlist.append(x);
        ylist.append(y);
    }
//    while(std::cin>>s){
//        qDebug()<<QString::fromStdString(s)<<'\n';
//        int pos = s.find(",");
//        double x = std::stod(s.substr(0,pos));
//        double y = std::stod(s.substr(pos+1));
//        qDebug()<<x<<" "<<y<<'\n';
//        xlist.append(x);
//        ylist.append(y);
//    }
    std::pair<QVector<double>,QVector<double>> ans = {xlist,ylist};
    return ans;
}

void ChartWidget::updateUI()
{
    qDebug()<<"进入到设置UI函数中-------------\n";
    QChart::ChartTheme theme = (QChart::ChartTheme)m_themeComboBox->itemData(m_themeComboBox->currentIndex()).toInt();
//    qDebug()<<chartView1<<'\n';
    if(chartView1->chart()->theme() != theme){
        chartView1->chart()->setTheme(theme);
        QPalette pal = window()->palette();
        if (theme == QChart::ChartThemeLight) {
//            pal.setColor(QPalette::Window, QRgb(0xf0f0f0));
            pal.setColor(QPalette::WindowText, QRgb(0x404044));
        } else if (theme == QChart::ChartThemeDark) {
//            pal.setColor(QPalette::Window, QRgb(0x121218));
            pal.setColor(QPalette::WindowText, QRgb(0xd6d6d6));
        } else if (theme == QChart::ChartThemeBlueCerulean) {
//            pal.setColor(QPalette::Window, QRgb(0x40434a));
            pal.setColor(QPalette::WindowText, QRgb(0xd6d6d6));
        } else if (theme == QChart::ChartThemeBrownSand) {
//            pal.setColor(QPalette::Window, QRgb(0x9e8965));
            pal.setColor(QPalette::WindowText, QRgb(0x404044));
        } else if (theme == QChart::ChartThemeBlueNcs) {
//            pal.setColor(QPalette::Window, QRgb(0x018bba));
            pal.setColor(QPalette::WindowText, QRgb(0x404044));
        } else if (theme == QChart::ChartThemeHighContrast) {
//            pal.setColor(QPalette::Window, QRgb(0xffab03));
            pal.setColor(QPalette::WindowText, QRgb(0x181818));
        } else if (theme == QChart::ChartThemeBlueIcy) {
//            pal.setColor(QPalette::Window, QRgb(0xcee7f0));
            pal.setColor(QPalette::WindowText, QRgb(0x404044));
        } else {
//            pal.setColor(QPalette::Window, QRgb(0xf0f0f0));
            pal.setColor(QPalette::WindowText, QRgb(0x404044));
        }
        window()->setPalette(pal);
    }
}

void ChartWidget::chooseCsvFile()
{
    qDebug()<<"进入到了ThemeWidget::chooseCsvFile函数----------\n";
    QString fileName = QFileDialog::getOpenFileName(this,QStringLiteral("选择文件"),"F:",QStringLiteral("excel文件(*csv)"));
    qDebug()<<"读取的文件是："<<fileName<<'\n';
    if(fileName.size() == 0){
        qDebug()<<"取消了选择文件\n";
        return ;
    }
    QFile file;
    file.setFileName(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return; // 打开失败
    std::pair<QVector<double>,QVector<double>> pos =  solveCsvFile(file);
    QVector<double> xlist = pos.first;
    QVector<double> ylist = pos.second;
//    if(chartView1 != nullptr) delete chartView1;
    chartView1->setChart(createLineChart(xlist,ylist));
//    chartView1 = new QChartView(createLineChart(xlist,ylist));
    chartView1->repaint();
    chartView1->update();
    chartView1->show();
//    QLineSeries *series =  (QLineSeries *)chartView1->chart()->series().at(0);
//    QVector<QPointF> pts = series->pointsVector();
//    for(int i = 0;i<pts.size();i++){
//        double x = pts[i].x();
//        double y = pts[i].y();
//        qDebug()<<x<<" "<<y<<'\n';
//    }
}
