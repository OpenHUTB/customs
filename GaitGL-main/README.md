1.按照requirements.txt安装环境。
2.在config.py文件中配置输入文件路径（底库）——dataset_path和测试文件路径——testset_path，路径中文件的存放格式按照output1（身份/角度/步态序列图片集合）和output2（身份/步态序列图片集合）中的文件存放格式来。
3.运行test1.py，可以得到每个测试数据对应的身份。
4.若路径中的文件数目发生变化要删除work/partition/CASIA-B_73_False.npy文件重新运行test1.py。
5.output1和output2中存放了少量数据用于测试。