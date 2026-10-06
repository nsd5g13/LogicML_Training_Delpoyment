#include <stdint.h>
#include <stddef.h>
//#include <stdio.h>
#include "layer0.h"
#include "layer1_uint32.h"
#include "samples.h"

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

int popcount32(int32_t x) {
    int count = 0;
    for (int i = 0; i < 32; i++) {
        if (x & 1) count++;
        x >>= 1;
    }
    return count;
}

// --- neuron operation ---
void neuron0(int32_t x[no_samples][ACTIVATIONS0], int w[ACTIVATIONS0], int32_t yp[no_samples],
            int N, int use_ste_sign) {

    int32_t tmp[no_samples];
    zeros1d(tmp, no_samples);

    for (int i = 0; i < N; i++) {               // iterate features
    	for (int j = 0; j < no_samples; j++) {  // iterate samples
            	//int32_t W = (w[i] == 0) ? -1 : 1;
            	//tmp[j] += x[j][i] * W;
		int32_t mult_result;
		mult_result = (w[i] == 0) ? 0-x[j][i] : x[j][i];
		tmp[j] += mult_result;
    	}
    }

    if (use_ste_sign) ste_sign(tmp, no_samples, 0);

    for (int i = 0; i < no_samples; i++) yp[i] = tmp[i];
}

// --- neuron operation ---
void neuron1(uint32_t x[no_samples][ACTIVATIONS1_PACKED], uint32_t w[ACTIVATIONS1_PACKED], int32_t yp[no_samples],
            int N, int use_ste_sign, int prev_N) {

    int32_t tmp[no_samples];
    zeros1d(tmp, no_samples);

    for (int i = 0; i < N; i++) {               // iterate features
    	for (int j = 0; j < no_samples; j++) {  // iterate samples
		int32_t bit_xnor = ~(x[j][i] ^ w[i]);
		int ones = popcount32(bit_xnor);
		tmp[j] += ones;
    	}
    }

    if (use_ste_sign) ste_sign(tmp, no_samples, prev_N/2);

    for (int i = 0; i < no_samples; i++) yp[i] = tmp[i];
}

// --- dense layer ---
void dense0(int32_t x[no_samples][ACTIVATIONS0], int w[][ACTIVATIONS0],
           int32_t res[][NEURONS0], int nunit, int N, int use_ste_sign) {
    int32_t tmp[no_samples];
    for (int i = 0; i < nunit; i++) {
        neuron0(x, w[i], tmp, N, use_ste_sign);
	for (int j = 0; j < no_samples; j++)
		res[j][i] = tmp[j];
    }
}

// --- dense layer ---
void dense1(uint32_t x[no_samples][ACTIVATIONS1_PACKED], uint32_t w[][ACTIVATIONS1_PACKED],
           int32_t res[][NEURONS1], int nunit, int N, int use_ste_sign, int prev_N) {
    int32_t tmp[no_samples];
    for (int i = 0; i < nunit; i++) {
        neuron1(x, w[i], tmp, N, use_ste_sign, prev_N);
	for (int j = 0; j < no_samples; j++)
		res[j][i] = tmp[j];
    }
}

// --- argmax for prediction ---
void argmax(int32_t x[][NEURONS1], int32_t labels[], int samples, int N) {
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

// yout: [no_samples][NEURONS0]
// packed: [no_samples][num_words_per_row]
// returns number of 32-bit words per row
int pack_bits(
    int32_t yout[][NEURONS0],
    uint32_t packed[][ACTIVATIONS1_PACKED],
    int NO_SAMPLES
) {
    int num_words = ACTIVATIONS1_PACKED;

    for (int row = 0; row < NO_SAMPLES; row++) {
        for (int w = 0; w < num_words; w++) {
            uint32_t word = 0;
            int start = w * 32;
            
            for (int bit = 0; bit < 32; bit++) {
                int col = start + bit;
                if (col < NEURONS0) {
                    // Insert bit (value must be 0 or 1)
                    word |= ( (uint32_t)(yout[row][col] & 1) << (31 - bit) );
                }
            }

            packed[row][w] = word;
        }
    }

    return num_words;
}

// --- main function ---
int32_t result_array[no_samples];

int main(void) {
    // Forward pass
    int32_t yout1[no_samples][NEURONS0];
    dense0(raw_samples, LAYER0, yout1, NEURONS0, ACTIVATIONS0, 1); // first layer: use_ste_sign=1
    uint32_t yout1_packed[no_samples][ACTIVATIONS1_PACKED];
    pack_bits(yout1, yout1_packed, no_samples);
    int32_t yout2[no_samples][NEURONS1];
    dense1(yout1_packed, LAYER1, yout2, NEURONS1, ACTIVATIONS1_PACKED, 0, NEURONS0); // second layer: no activation

    argmax(yout2, result_array, no_samples, NEURONS1);

    // For debugging only
    //for (int i = 0; i < no_samples; i++) {
    //    printf("%d ", result_array[i]);
    //}
    //printf("\n");

    // labels now contain the predicted class for each sample
    //while(1); // bare-metal infinite loop
    return 0;
}
