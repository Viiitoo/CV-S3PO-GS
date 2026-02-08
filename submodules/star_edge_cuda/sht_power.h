#ifndef SHT_POWER_H
#define SHT_POWER_H

void power_spectrum_kernel(
    int N, int bw,
    const float* coeffs_real,   // (N, bw*bw)
    const float* coeffs_imag,   // (N, bw*bw)
    float* descriptors);        // (N, bw)

#endif
