#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "chartwidget.h"
#include <QWidget>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setCentralWidget(ui->tabWidget);


    ChartWidget *widget = new ChartWidget(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("Doc %d",ui->tabWidget->count()));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_ShowChart_triggered()
{
    ChartWidget *widget = new ChartWidget(this);
    widget->setAttribute(Qt::WA_DeleteOnClose);
    int cur = ui->tabWidget->addTab(widget,QString::asprintf("Doc %d",ui->tabWidget->count()));
    ui->tabWidget->setCurrentIndex(cur);
    ui->tabWidget->setVisible(true);
}
