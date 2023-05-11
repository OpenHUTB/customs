1、安装PaddlePaddle和PaddleDetection

2、按照requirements.txt安装环境

3、模型文件链接如下：链接：https://pan.baidu.com/s/1qhPhq0bwAENLkWpJ6KDM8g  提取码：07bi

4、下载模型并解压到./output_inference路径下（output_inference为新建文件夹）

5、图片输入时，终端启动命令如下：

```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --image_file=test_image.jpg --device=gpu --enable_attr=True [--run_mode trt_fp16]
```

6、视频输入时，终端启动命令如下：

```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=test_video.mp4 --device=gpu --enable_attr=True [--run_mode trt_fp16]
```

7、若修改模型路径，有以下两种方式：

```python
  - ```./deploy/pphuman/config/infer_cfg.yml```下可以配置不同模型路径，属性识别模型修改ATTR字段下配置
    - 命令行中增加`--model_dir`修改模型路径：
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \ --video_file=test_video.mp4 \ --device=gpu \--enable_attr=True \ --model_dir det=ppyoloe/
```

8、技术文件夹链接如下：链接：https://pan.baidu.com/s/1OkoM2RhBAteoSkGEEeU0lg 提取码：gx2u

9、效果输出图示例文件夹链接如下：链接：https://pan.baidu.com/s/1h2gZRERhQIeCPuyukL4C4w 提取码：ohgl

10、配置文件文件夹链接如下：链接：https://pan.baidu.com/s/1H8KqVnCd7cHDheMxEYuJtA 
提取码：qp4z



