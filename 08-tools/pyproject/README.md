# 基于pyproject的项目管理

在之前的项目管理中，我们往往使用setup.py, requirements.txt, setup.cfg等来管理项目。整体的流程较为繁琐，而pyproject 将这些整合到一起
 

如下所示，首先会有一个`build_system`，其会指定编译的后端以及需要的工具

```shell
[build-system]
# 编译时依赖：nanobind 和 scikit-build-core
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"

[project]
name = "my_ext"
version = "0.0.1"

# 配置 scikit-build-core
[tool.scikit-build]
# 指定 CMakeLists.txt 所在的目录
cmake.source-dir = "."
```

然后是project的名称和版本等元数据

```shell
[project]
name = "your-awesome-project"
version = "0.1.0"
description = "一个很棒的项目"
authors = [
    {name = "作者名", email = "author@example.com"}
]
dependencies = [
    "requests>=2.24.0",
    "pandas>=1.0.0"
]
```

开发依赖和可选功能，如下所示为dev和docs不同场景设置不同的包依赖

```toml
[project.optional-dependencies]
dev = ["pytest", "black"]
docs = ["sphinx"]
# 'all' 包含了 dev 和 docs 的所有内容
all = ["my-project[dev,docs]"]
```

配置其他所需工具的配置

```toml
[tool.black]
line-length = 88
target-version = ['py37']

[tool.isort]
profile = "black"

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q" 解释一下这部分是什么
```