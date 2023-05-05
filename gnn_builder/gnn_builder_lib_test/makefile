VITIS_HLS_ROOT := /tools/software/xilinx/Vitis_HLS/2022.1
BUILD_SRC_ROOT := .

IFLAG += -I "${VITIS_HLS_ROOT}/include"

CFLAG += -L"${VITIS_HLS_ROOT}/lnx64/lib/csim" -lhlsmc++-CLANG39 -Wl,-rpath,"${VITIS_HLS_ROOT}/lnx64/lib/csim" -Wl,-rpath,"${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_0"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

IFLAG += -I "${BUILD_SRC_ROOT}"

IFLAG += -g
CFLAG += -fPIC -fPIE -O3 -Wno-unused-result
CFLAG += -lm
CFLAG += -std=c++14

# CFLAG += -fsanitize=address -fno-omit-frame-pointer

# CXX = g++
# CXX = clang
CXX = ${VITIS_HLS_ROOT}/lnx64/tools/clang-3.9/bin/clang++


./result: ./*.cpp ./*.h ./../gnn_builder_lib/*.h
	$(CCACHE) $(CXX) ${BUILD_SRC_ROOT}/*.cpp -o result $(CFLAG) $(IFLAG)

all:
	./result

run: ./result
	./result

clean:
	rm -f *.o result

.PHONY: all run clean