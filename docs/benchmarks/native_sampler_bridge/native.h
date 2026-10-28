#pragma once
#include <cstddef>
#include <cstdint>
// Standalone diagnostic integration. All arrays are contiguous row-major.
// cfg: engine(0 Stan,1 Walnutpie), kind(0..3), D,C,W,S,depth,halvings.
// controls: bend, initial_step_size, target_accept, Walnutpie energy bound.
// rows [C,W+S,D+7]: native lp,accept,epsilon,depth,nleap,divergent,energy,q.
// Unsupported Walnutpie diagnostics are NaN (not zero).
// meta [C,D+4]: epsilon, inverse_mass[D], min_micro_steps, grad_calls, exceptions.
extern "C" int bf_native_fit(const int64_t* cfg, const double* controls,
    const double* starts, const double* inverse_transform, const int64_t* seeds,
    double* rows, double* meta, char* error, size_t error_size) noexcept;
extern "C" int bf_native_gradient(int64_t kind, int64_t dim, double bend,
    const double* inverse_transform, const double* q, double* lp_grad,
    char* error, size_t error_size) noexcept;
