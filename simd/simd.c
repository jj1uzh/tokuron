#include<arm_neon.h>
#include<stdbool.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

// 行列のサイズ
#define SIZE 3001
#define SIZE2 SIZE*SIZE

// 行を4つ組のSequenceに変換した際のサイズ
#define SIZE_F4 (((SIZE) >> 2) + 1)

// 計測マクロ
static clock_t clk_bgn, clk_end;
static char* lap_name;
#define LAP_START(name)                         \
        clk_bgn = clock();                      \
        lap_name = name
#define LAP_END()                                                       \
        clk_end = clock();                                              \
        fprintf(stderr, "[%s] %2.4f\n", lap_name, (double)(clk_end - clk_bgn) / CLOCKS_PER_SEC)

// floatの4つ組
struct f4 {
        float32_t _[4];
};
typedef struct f4 f4_t;

void mrx_rand(float *m)
{
        for (int i = 0; i < SIZE * SIZE; i++, m++) {
                *m = rand() % 10;
        }
}

void mrx_print(float *m)
{
        for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++, m++) {
                        printf("%9.4f ", *m);
                }
                printf("\n");
        }
        printf("\n");
}

void mrx_transpose(float *m)
{
        for (int i = 0; i < SIZE; i++) {
                for (int j = i; j < SIZE; j++) {
                        float tmp = m[i * SIZE + j];
                        m[i * SIZE + j] = m[j * SIZE + i];
                        m[j * SIZE + i] = tmp;
                }
        }
}

// 変換

// 行を f4_t の配列で表わした行列に変換
void mrx_to_f4mrx(f4_t *dst, float *src)
{
        f4_t v = { ._ = {0, 0, 0, 0} };
        for (int i = 0; i < SIZE; i++) {
                int j;
                for (j = 0; j < SIZE_F4 - 1; j++) {
                        dst[i * SIZE_F4 + j] = *(f4_t *)(src + (i * SIZE) + (j << 2));
                }
                memcpy(v._, src + (i * SIZE) + (j << 2), sizeof(float) * (SIZE % 4));
                dst[(i + 1) * SIZE_F4 - 1] = v;
        }
}

// f4_tで表した行列をfloat32x4_tのものに変換
void f4mrx_to_simdmrx(float32x4_t *dst, f4_t *src)
{
        for (int i = 0; i < SIZE * SIZE_F4; i++) {
                dst[i] = vld1q_f32(src[i]._);
        }
}

// 行列積の実装

void mrx_mult_nosimd_1(float *dst, f4_t *m0, f4_t *m1)
{
        f4_t *m1_head = m1;
        LAP_START("nosimd_1");
        for (int i = 0; i < SIZE;
             i++, m0 += SIZE_F4, m1 = m1_head)
        {
                for (int j = 0; j < SIZE;
                     j++, dst++, m0 -= SIZE_F4)
                {
                        float sum = 0;
                        for (int k = 0; k < SIZE_F4;
                             k++, m0++, m1++)
                        {
                                f4_t prod, v0 = *m0, v1 = *m1;
#define mul(i) prod._[i] = v0._[i] * v1._[i]
                                mul(0); mul(1); mul(2); mul(3);
                                sum += prod._[0] + prod._[1] + prod._[2] + prod._[3];
                        }
                        *dst = sum;
                }
        }
        LAP_END();
}

void mrx_mult_nosimd_2(float *dst, f4_t *m0, f4_t *m1)
{
        f4_t *m1_head = m1;
        LAP_START("nosimd_2");
        for (int i = 0; i < SIZE;
             i++, m0 += SIZE_F4, m1 = m1_head)
        {
                for (int j = 0; j < SIZE;
                     j++, dst++, m0 -= SIZE_F4)
                {
                        float sum = 0;
                        for (int k = 0; k < SIZE_F4;
                             k++, m0++, m1++)
                        {
                                f4_t prod, v0 = *m0, v1 = *m1;
                                mul(0);
                                sum += prod._[0];
                                mul(1);
                                sum += prod._[1];
                                mul(2);
                                sum += prod._[2];
                                mul(3);
                                sum += prod._[3];
                        }
                        *dst = sum;
                }
        }
        LAP_END();
}

void mrx_mult_simd(float *dst, float32x4_t *m0, float32x4_t *m1)
{
        float32x4_t *m1_head = m1;
        LAP_START("simd");
        for (int i = 0; i < SIZE;
             i++, m0 += SIZE_F4, m1 = m1_head)
        {
                for (int j = 0; j < SIZE;
                     j++, dst++, m0 -= SIZE_F4)
                {
                        float sum = 0;
                        for (int k = 0; k < SIZE_F4;
                             k++, m0++, m1++)
                        {
                                float32x4_t __attribute__ ((aligned(32))) prod = vmulq_f32(*m0, *m1);
                                sum += vaddvq_f32(prod);
                        }
                        *dst = sum;
                }
        }
        LAP_END();
}

int main(int argc, char** argv __attribute__((unused)))
{
        bool debugp = argc > 1;
#define debug(m) if (debugp) { mrx_print((m)); }
        printf("SIZE=%d\n", SIZE);

        // 準備
        static float mrx0[SIZE2], mrx1[SIZE2];
        mrx_rand(mrx0);
        mrx_rand(mrx1);
        debug(mrx0);
        debug(mrx1);
        mrx_transpose(mrx1);

        static f4_t f4mrx0[SIZE * SIZE_F4], f4mrx1[SIZE * SIZE_F4];
        static float dst_f4_1[SIZE2], dst_f4_2[SIZE2];
        mrx_to_f4mrx(f4mrx0, mrx0);
        mrx_to_f4mrx(f4mrx1, mrx1);

        static float32x4_t
                __attribute__ ((aligned(32))) simdmrx0[SIZE * SIZE_F4],
                __attribute__ ((aligned(32))) simdmrx1[SIZE * SIZE_F4];
        static float dst_simd[SIZE2];
        f4mrx_to_simdmrx(simdmrx0, f4mrx0);
        f4mrx_to_simdmrx(simdmrx1, f4mrx1);

        // 計測
        mrx_mult_nosimd_1(dst_f4_1, f4mrx0, f4mrx1);
        debug(dst_f4_1);

        mrx_mult_nosimd_2(dst_f4_2, f4mrx0, f4mrx1);
        debug(dst_f4_2);

        mrx_mult_simd(dst_simd, simdmrx0, simdmrx1);
        debug(dst_simd);

        return 0;
}
