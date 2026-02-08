#ifndef SPHERE_KDE_H
#define SPHERE_KDE_H

void upload_grid_to_constant(const float* grid_xyz, int G);

void sphere_kde_fused(
    int N, int S, int G,
    float h,
    const float* samples,      // (N, S, 3)
    float* sphere_func);       // (N, G)

#endif
