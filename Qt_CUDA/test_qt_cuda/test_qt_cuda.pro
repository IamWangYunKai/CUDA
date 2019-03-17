QT       += core
QT       -= gui

TARGET = TestCUDA
CONFIG   += console

DESTDIR = release
OBJECTS_DIR = release/obj
CUDA_OBJECTS_DIR = release/cuda

# Source files
SOURCES += main.cpp

# This makes the .cu files appear in your project
OTHER_FILES +=  kernel.cu

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += kernel.cu
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0"           # Path to cuda toolkit install
SYSTEM_NAME = x64           # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
# Type of CUDA architecture
#61:GTX 1080, GTX 1070, GTX 1060, GTX 1050, Titan Xp
#62:Tegra (Jetson) TX2
#Volta (CUDA 9 and later):
#70:GTX 1180 (GV104), Titan V
#Turing (CUDA 10 and later):
#75:GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX
CUDA_ARCH = compute_61
CUDA_CODE = sm_61
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME

# The following library conflicts with something in Cuda
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib

MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CUDA_LIB_NAMES += cuda cudart MSVCRT

for(lib, CUDA_LIB_NAMES) {
    CUDA_LIBS += $$lib.lib
}
for(lib, CUDA_LIB_NAMES) {
    NVCC_LIBS += -l$$lib
}
LIBS += $$NVCC_LIBS

cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS \
                --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -code=$$CUDA_CODE \
                --compile -cudart static \
                -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda

