#include <stdint.h>
#include <stddef.h>
//#include <stdio.h>
#include "layer0.h"
#include "layer1.h"
#include "samples.h"
#include "helpers.h"

// --- main function ---
int32_t result_array[no_samples];

int main(void) {
    // Forward pass
    int32_t yout1[no_samples][N_MAX];
    dense(raw_samples, LAYER0, yout1, NEURONS0, ACTIVATIONS0, 1, 1); // first layer: InputLayer=True, use_ste_sign=1
    int32_t yout2[no_samples][N_MAX];
    dense(yout1, LAYER1, yout2, NEURONS1, ACTIVATIONS1, 0, 0); // second layer: InputLayer=False, no activation

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
