#include "helpers.h"

// --- Helper functions ---
void add1d(const int32_t a[], const int32_t b[], int32_t out[], int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void zeros1d(int32_t arr[], int n) {
    for (int i = 0; i < n; i++) arr[i] = 0;
}

void ste_sign(int32_t arr[], int n, int32_t zero_value) {
    for (int i = 0; i < n; i++) {
        if (arr[i] > zero_value) arr[i] = 1;
        else arr[i] = 0;
    }
}

// --- neuron operation ---
void neuron(int32_t x[no_samples][N_MAX], int w[N_MAX], int32_t yp[no_samples],
            int N, int InputLayer, int use_ste_sign) {

    int32_t tmp[no_samples];
    zeros1d(tmp, no_samples);

    for (int i = 0; i < N; i++) {               // iterate features
    	for (int j = 0; j < no_samples; j++) {  // iterate samples
        	if (InputLayer) {
            		//int32_t W = (w[i] == 0) ? -1 : 1;
            		//tmp[j] += x[j][i] * W;
			int32_t mult_result;
			mult_result = (w[i] == 0) ? 0-x[j][i] : x[j][i];
			tmp[j] += mult_result;
        	} else {
            		tmp[j] += (x[j][i] == w[i]) ? 1 : 0;
        	}
    	}
    }

    if (use_ste_sign) ste_sign(tmp, no_samples, 0);

    for (int i = 0; i < no_samples; i++) yp[i] = tmp[i];
}

// --- dense layer ---
void dense(int32_t (*x)[N_MAX], int w[][N_MAX],
           int32_t (*res)[N_MAX], int nunit, int N, int InputLayer, int use_ste_sign) {
    int32_t tmp[no_samples];
    for (int i = 0; i < nunit; i++) {
        neuron(x, w[i], tmp, N, InputLayer, use_ste_sign);
	for (int j = 0; j < no_samples; j++)
		res[j][i] = tmp[j];
    }
}

// --- argmax for prediction ---
void argmax(int32_t (*x)[N_MAX], int32_t labels[], int samples, int N) {
    for (int j = 0; j < samples; j++) {
        int32_t max_val = x[0][j];
        int label = 0;
        for (int i = 1; i < N; i++) {
            if (x[i][j] > max_val) {
                max_val = x[i][j];
                label = i;
            }
        }
        labels[j] = label;
    }
}
