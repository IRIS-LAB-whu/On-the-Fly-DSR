# phantom3 小型示例

本目录包含维护者指定的 `D:/Dataset/CAS/phantom3/50` 子集，共 50 张原始 JPG（约 51 MiB）。仅复制影像，未缩放、重编码或加入重建产物。逐文件校验值见 images.sha256。完整 images 目录和 phantom3.zip 未随仓库重复分发。

## 运行

先完成 README 中的环境、权重下载及 Release/x64 构建。在运行时 settings.ini 中确认 PythonHome / PythonWorkspace 正确。将下例路径替换为自己的仓库绝对路径，并预先创建 output 目录：

~~~ini
[Pipeline]
AutoRunImages=true

[Experiment]
Enable=true
ProjectPath=D:/your-path/On-the-FlyDSR/examples/phantom3/output/phantom3.db
ImagePath=D:/your-path/On-the-FlyDSR/examples/phantom3/images
ShowUI=true
AutoSaveIdleSeconds=60
~~~

以上仅展示需要修改的字段，其他参数保留完整配置中的值。将 Experiment 中对应字段替换，避免新增重复 section。每次重新测试使用新的空输出目录和新的数据库文件。

从 UI.exe 所在目录启动。预期逐步出现已注册相机与稀疏点；默认 SupScene + DepthAnythingV2 配置在注册数量达到阈值后开始深度估计、对齐与密集融合。数据耗尽且工作流空闲后自动导出，查看输出目录、UI 日志以及 DenseReconstruction 下的结果。注册数量、耗时、点数和覆盖率取决于参数与硬件，未承诺固定数值。

维护者报告已完成 phantom3 实际重建验证；这不等于已独立验证当前 50 张子集或在全新机器上复现。当前打包检查覆盖文件数量、大小与 SHA-256。

## 数据来源与分发

影像由项目维护者提供并指定作为发布示例。代码的 GPL/BSD 许可证不自动适用于照片；本次未收到照片的独立许可文本。正式公开发布前应补充数据来源、版权人及允许使用/再分发的条款。本文件不虚构数据许可证。
