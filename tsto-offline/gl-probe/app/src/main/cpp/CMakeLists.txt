cmake_minimum_required(VERSION 3.18.1)
project(scorpio_glprobe LANGUAGES CXX)
add_library(glprobe SHARED probe.cpp)
target_link_libraries(glprobe PRIVATE GLESv2 log)
