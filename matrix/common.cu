//バッファだけ
//エスケープシーケンス用です。
#include <stdarg.h>
#include "common.h"
#include "string.h"

char _esc_seq_buf[1024];


__host__ __device__ void pos_printf(int x, int y, char* fmt,...) {
    char buf[256];
    sprintf_s(buf,sizeof(buf), "\x1B[%d;%dH%s", y, x,fmt);
    va_list ap;    va_start(ap, fmt);
    vprintf_s(buf, ap);
//    printf("\x1B[%d;%dH", y, x );
}
__host__ __device__ void esc_clr()
{
    printf("\x1B[2J");
}
