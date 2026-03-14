cmake_minimum_required(VERSION 3.15)
project(my_ext_project)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(my_ext my_ext.cpp)

# ========================================================
# 【关键修复】必须添加这行安装指令
# 告诉 CMake 把生成的 my_ext 模块安装到 Python 包的根目录
# ========================================================
install(TARGETS my_ext DESTINATION .)