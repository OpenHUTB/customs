#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "chartwidget.h"
#include "siglechartwidget.h"
#include "monthpredictchartwidget.h"
#include "PerDayError.h"
#include "erroronday.h"
#include <QWidget>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->tabWidget->setVisible(false);

//    ChartWidget *widget = new ChartWidget(this);
//    widget->setAttribute(Qt::WA_DeleteOnClose);
//    int cur = ui->tabWidget->addTab(widget,QString::asprintf("Doc %d",ui->tabWidget->count()));
//    ui->tabWidget->setCurrentIndex(cur);
//    ui->tabWidget->setVisible(true);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_ShowChart_triggered()
{
    init();
    ChartWidget *widget = new ChartWidget(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("Doc %d",ui->tabWidget->count()));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}


void MainWindow::on_action_triggered()
{
    init();
    SigleChartWidget *widget = new SigleChartWidget(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("模型训练图"));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}

void MainWindow::on_action_2_triggered()
{
    init();
    MonthPredictChartWidget *widget = new MonthPredictChartWidget(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("月度用电量预测对比"));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}

void MainWindow::on_action_3_triggered()
{
    init();
    PerDayError *widget = new PerDayError(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("日用电量对比曲线"));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}

void MainWindow::on_action_4_triggered()
{
    init();
    ErrorOnDay *widget = new ErrorOnDay(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("日用电量准确率曲线"));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}

void MainWindow::init()
{
    this->setCentralWidget(ui->tabWidget);
}
