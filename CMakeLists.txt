cmake_minimum_required(VERSION 3.20)
project(leetcode_timu LANGUAGES CXX)

# ================= C++ 标准 =================
set(CMAKE_CXX_STANDARD 20)          # GlobalLRU_cache 需要 C++20 的 std::atomic<std::shared_ptr>
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 未指定构建类型时默认 Debug(单配置生成器需要)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif ()

# 导出 compile_commands.json,供 clangd / clang-tidy 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ================= 依赖 =================
# std::mutex:Linux/WSL 下需要链接 pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---- CUDA(可选)----
# leetcode_timu 需要 CUDA Toolkit;test_global_lru 是纯 C++ 测试,不需要。
# 注意:Windows 上 nvcc 只支持 MSVC 工具链,MinGW 配不了 CUDA。
include(CheckLanguage)
option(ENABLE_CUDA "Build the CUDA demo target (requires CUDA Toolkit)" ON)
if (ENABLE_CUDA)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)   # 走到这里说明 nvcc 存在,提供 CUDA::cudart
        # 按你的 GPU 调整;不设的话新版 CMake 会因 CMP0104 报错
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 86 90)
        endif ()
    else ()
        message(WARNING
            "CUDA compiler not found - 'leetcode_timu' target will be skipped. "
            "Install the CUDA Toolkit. On Windows use the MSVC toolchain, not MinGW.")
    endif ()
endif ()

# ---- TensorRT(可选,默认关闭)----
# 只有当你真正写了 TensorRT 代码再打开 ENABLE_TENSORRT。
# 注意:在这台机器上(CUDA 13.3 + WSL + TensorRT 11.2),只要把 TensorRT 的库
# 链进一个会启动 CUDA kernel 的可执行文件,内核启动时的 JIT 阶段就会段错误
# (libnvidia-ptxjitcompiler 崩溃)。leetcode_timu 目前没有任何 TensorRT 调用,
# 所以默认不链接——等真用到时再开,并留意这个环境问题。
option(ENABLE_TENSORRT "Link TensorRT into leetcode_timu (only if the code actually uses it)" OFF)
if (ENABLE_TENSORRT)
    find_package(TensorRT CONFIG REQUIRED)
    message(STATUS "TensorRT found: ${TensorRT_VERSION}")
endif ()

# ================= 目标 =================
# 头文件加入目标只是为了在 CLion 项目树里可见
file(GLOB_RECURSE LEETCODE_HEADERS CONFIGURE_DEPENDS "include/*.h" "include/*.hpp")

# 纯 C++ 测试,不依赖 CUDA / TensorRT
add_executable(test_global_lru test_global_lru.cc ${LEETCODE_HEADERS})
target_include_directories(test_global_lru PRIVATE include)
target_link_libraries(test_global_lru PRIVATE Threads::Threads)

# CUDA 演示程序
if (CMAKE_CUDA_COMPILER)
    add_executable(leetcode_timu main.cu src/cuda_wrapper.cu ${LEETCODE_HEADERS})
    target_include_directories(leetcode_timu PRIVATE include)
    # 注意:不要显式链接 CUDA::cudart!CMake 的设备链接(见下)已经默认链了静态的
    # libcudart_static;再把 CUDA::cudart(动态 libcudart.so)加进来,进程里就有两份
    # CUDA 运行时,内核启动走 JIT 时会段错误(__cuda_CallJitEntryPoint 崩溃,在
    # RTX 4060 / CUDA 13.3 / WSL 上稳定复现)。cudart 由 CMake 自动处理即可。
    target_link_libraries(leetcode_timu PRIVATE Threads::Threads)

    # 让 CMake 在链接期做设备链接(走 nvcc),而不是用 g++ 把 fatbin 原样拼进
    # 可执行文件。否则跨编译单元的设备符号(device printf 的桩)只能在运行时
    # 靠 JIT 现场合并——某些 CUDA/驱动组合下这一步会崩溃。
    set_target_properties(leetcode_timu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    if (ENABLE_TENSORRT)
        target_link_libraries(leetcode_timu PRIVATE
                TensorRT::nvinfer
                TensorRT::nvinfer_plugin
                TensorRT::nvonnxparser)
    endif ()
endif ()

# ================= 编译告警 =================
# 只对 C++ 编译应用:nvcc 不认识 -Wshadow 这类 host 选项。
# (leetcode_timu 的源文件全是 .cu,走 CUDA 语言,这里只命中 test_global_lru。)
if (MSVC)
    target_compile_options(test_global_lru PRIVATE /W4 /utf-8)
else ()
    target_compile_options(test_global_lru PRIVATE
            -Wall -Wextra -Wpedantic
            -Wshadow                 # 变量遮蔽
            -Wno-unused-parameter
    )
endif ()

# ================= 可选:Sanitizer(排查越界 / UB)=================
# 只作用于纯 C++ 的测试目标;CUDA + ASan 组合复杂,需要时再单独处理。
option(ENABLE_SANITIZERS "Enable Address/UB sanitizers" OFF)
if (ENABLE_SANITIZERS AND NOT MSVC AND NOT MINGW)
    target_compile_options(test_global_lru PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(test_global_lru PRIVATE -fsanitize=address,undefined)
endif ()
