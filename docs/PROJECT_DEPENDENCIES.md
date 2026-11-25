# 项目生成依赖与 Visual Studio 源码浏览

## 已保存的生成依赖

依赖保存在两处：Workflow/UI 的 .vcxproj ProjectReference，以及 OntheFlySfM.sln 的 ProjectSection(ProjectDependencies)。两处保持相同关系。

| 项目 | 直接生成依赖 |
| --- | --- |
| Base、Geometry、Scene、Estimator、Feature、SIFTGPU、VLFeat | 当前没有显式生成依赖，可独立生成静态库；这不表示源码没有头文件或符号引用 |
| Open3DWeightedTSDF | 通过 Makefile 项目调用 Open3D 自身的 CMake 工程 |
| Workflow | Base、Geometry、Scene、Estimator、Feature、SIFTGPU、VLFeat、Open3DWeightedTSDF |
| UI | Workflow 的上述 8 项依赖，以及 Workflow 本身 |

底层模块存在交叉头文件引用，不能把所有 #include 机械转换为生成依赖，否则会引入循环。静态库归档阶段不要求解析全部外部符号；UI 最终链接各库。Open3D 的引用明确设置 LinkLibraryDependencies=false，其 Open3D.lib 仍通过 UI 链接配置引用。

验证：解决方案能解析 Workflow 的 8 项、UI 的 9 项依赖，无循环及未知 GUID。

## 在 Visual Studio 中查看

1. 打开 OntheFlySfM.sln；若已打开，收到外部修改提示时重新加载项目/解决方案。
2. 在解决方案资源管理器中右键 Workflow 或 UI，选择“生成依赖项 → 项目依赖项”；“项目生成顺序”页可查看自动安排的顺序。
3. 项目引用产生的依赖可能显示为不可取消的勾选，这是引用驱动的依赖。要修改此类关系应修改项目引用，而非仅编辑解决方案顺序。
4. 需要使用解决方案视图；仅用“打开文件夹”进入 CMake 文件夹视图时，菜单和依赖展示会不同。

## Open3D 源码显示

Open3DWeightedTSDF 是一个 Makefile 构建入口，不是手工维护每个 Open3D 编译单元的普通 C++ 库工程。现在 .vcxproj 列出了源码，.vcxproj.filters 按原路径分组，源码可在 Open3D/cpp/open3d 等节点下展开。837 个 C/C++ 文件、581 个头文件、71 个 CUDA 文件及相应 CMake 文件已加入列表；生成仍交给 CMake，不会由包装工程重复编译。

新增或删除 Open3D 源文件时，应同步更新工程文件和 filters 列表。深度浏览/编辑后，CMake 在下一次生成时判断实际重编译范围。
