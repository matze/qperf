all: qperf

qperf: qperf.c
	gcc -I/usr/local/cuda/include `pkg-config --libs --cflags glib-2.0` -lOpenCL --std=c99 -o qperf qperf.c

