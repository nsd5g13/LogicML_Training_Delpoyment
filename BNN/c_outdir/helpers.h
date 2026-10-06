
#ifndef HELPERS_H
#define HELPERS_H

#include "layer0.h"
#include "layer1.h"
#include "samples.h"

void add1d(const int32_t a[], const int32_t b[], int32_t out[], int n);
void zeros1d(int32_t arr[], int n);
void ste_sign(int32_t arr[], int n, int32_t zero_value);
void neuron(int32_t x[no_samples][N_MAX], int w[N_MAX], int32_t yp[no_samples],
            int N, int InputLayer, int use_ste_sign);
void dense(int32_t (*x)[N_MAX], int w[][N_MAX],
           int32_t (*res)[N_MAX], int nunit, int N, int InputLayer, int use_ste_sign);
void argmax(int32_t (*x)[N_MAX], int32_t labels[], int samples, int N);

#endif