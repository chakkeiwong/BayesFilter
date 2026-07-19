#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"

#ifdef HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

#include "tensorflow/compiler/xla/hlo/builder/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

bool has_nonfinite(const double* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (!std::isfinite(data[i])) return true;
  }
  return false;
}

double max_abs_symmetry_residual(const double* s, int n, double* max_abs) {
  double residual = 0.0;
  double scale = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double a = s[i * n + j];
      const double b = s[j * n + i];
      residual = std::max(residual, std::abs(a - b));
      scale = std::max(scale, std::abs(a));
    }
  }
  *max_abs = scale;
  return residual;
}

int solve_one(int n, const double* s, const double* rhs, double* x,
              std::string* error) {
  const size_t nn = static_cast<size_t>(n) * n;
  if (has_nonfinite(s, nn) || has_nonfinite(rhs, nn)) {
    *error = "SymmetricSylvester: inputs must be finite";
    return 1;
  }

  double max_abs = 0.0;
  const double sym_residual = max_abs_symmetry_residual(s, n, &max_abs);
  const double sym_tolerance = 1.0e-10 * (1.0 + max_abs);
  if (sym_residual > sym_tolerance) {
    std::ostringstream oss;
    oss << "SymmetricSylvester: symmetric_factor must be symmetric; residual="
        << sym_residual << " tolerance=" << sym_tolerance;
    *error = oss.str();
    return 2;
  }

  Eigen::Map<const RowMajorMatrix> s_map(s, n, n);
  Eigen::Map<const RowMajorMatrix> rhs_map(rhs, n, n);
  const Eigen::MatrixXd s_col = s_map;
  const Eigen::MatrixXd rhs_col = rhs_map;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(s_col);
  if (eig.info() != Eigen::Success) {
    *error = "SymmetricSylvester: eigensolve failed";
    return 3;
  }
  const Eigen::VectorXd values = eig.eigenvalues();
  const double min_value = values.minCoeff();
  if (!(min_value > 0.0)) {
    std::ostringstream oss;
    oss << "SymmetricSylvester: symmetric_factor must be positive definite; "
        << "min_eigenvalue=" << min_value;
    *error = oss.str();
    return 4;
  }

  const Eigen::MatrixXd vectors = eig.eigenvectors();
  Eigen::MatrixXd transformed = vectors.transpose() * rhs_col * vectors;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      transformed(i, j) /= values(i) + values(j);
    }
  }
  const Eigen::MatrixXd solution = vectors * transformed * vectors.transpose();
  Eigen::Map<RowMajorMatrix> x_map(x, n, n);
  x_map = solution;
  return 0;
}

int solve_batch(int batch, int parameters, int n, const double* s,
                const double* rhs, double* x, std::string* error) {
  const size_t nn = static_cast<size_t>(n) * n;
  for (int b = 0; b < batch; ++b) {
    const double* s_b = s + static_cast<size_t>(b) * nn;
    for (int p = 0; p < parameters; ++p) {
      const size_t offset = (static_cast<size_t>(b) * parameters + p) * nn;
      const int info = solve_one(n, s_b, rhs + offset, x + offset, error);
      if (info != 0) return info;
    }
  }
  return 0;
}

bool parse_opaque(const char* opaque, size_t opaque_len, int* batch,
                  int* parameters, int* n, std::string* error) {
  try {
    const std::string text(opaque, opaque_len);
    const size_t comma1 = text.find(',');
    const size_t comma2 = text.find(',', comma1 == std::string::npos ? 0 : comma1 + 1);
    if (comma1 == std::string::npos || comma2 == std::string::npos) {
      throw std::runtime_error("missing comma");
    }
    *batch = std::stoi(text.substr(0, comma1));
    *parameters = std::stoi(text.substr(comma1 + 1, comma2 - comma1 - 1));
    *n = std::stoi(text.substr(comma2 + 1));
  } catch (...) {
    *error = "SymmetricSylvesterXlaImpl: cannot parse opaque shape";
    return false;
  }
  return *batch > 0 && *parameters > 0 && *n > 0;
}

int principal_sqrt_one(int n, const double* covariance, double* factor,
                       std::string* error) {
  const size_t nn = static_cast<size_t>(n) * n;
  if (has_nonfinite(covariance, nn)) {
    *error = "SymmetricPrincipalSqrt: covariance must be finite";
    return 1;
  }

  double max_abs = 0.0;
  const double sym_residual =
      max_abs_symmetry_residual(covariance, n, &max_abs);
  const double sym_tolerance = 1.0e-10 * (1.0 + max_abs);
  if (sym_residual > sym_tolerance) {
    std::ostringstream oss;
    oss << "SymmetricPrincipalSqrt: covariance must be symmetric; residual="
        << sym_residual << " tolerance=" << sym_tolerance;
    *error = oss.str();
    return 2;
  }

  Eigen::Map<const RowMajorMatrix> covariance_map(covariance, n, n);
  const Eigen::MatrixXd covariance_col = covariance_map;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance_col);
  if (eig.info() != Eigen::Success) {
    *error = "SymmetricPrincipalSqrt: eigensolve failed";
    return 3;
  }
  const Eigen::VectorXd values = eig.eigenvalues();
  const double min_value = values.minCoeff();
  if (!(min_value > 0.0)) {
    std::ostringstream oss;
    oss << "SymmetricPrincipalSqrt: covariance must be positive definite; "
        << "min_eigenvalue=" << min_value;
    *error = oss.str();
    return 4;
  }

  const Eigen::MatrixXd vectors = eig.eigenvectors();
  const Eigen::MatrixXd sqrt_diag = values.array().sqrt().matrix().asDiagonal();
  const Eigen::MatrixXd sqrt_matrix =
      vectors * sqrt_diag * vectors.transpose();
  Eigen::Map<RowMajorMatrix> factor_map(factor, n, n);
  factor_map = sqrt_matrix;
  return 0;
}

int principal_sqrt_batch(int batch, int n, const double* covariance,
                         double* factor, std::string* error) {
  const size_t nn = static_cast<size_t>(n) * n;
  for (int b = 0; b < batch; ++b) {
    const size_t offset = static_cast<size_t>(b) * nn;
    const int info =
        principal_sqrt_one(n, covariance + offset, factor + offset, error);
    if (info != 0) return info;
  }
  return 0;
}

bool parse_matrix_opaque(const char* opaque, size_t opaque_len, int* batch,
                         int* n, std::string* error) {
  try {
    const std::string text(opaque, opaque_len);
    const size_t comma = text.find(',');
    if (comma == std::string::npos) {
      throw std::runtime_error("missing comma");
    }
    *batch = std::stoi(text.substr(0, comma));
    *n = std::stoi(text.substr(comma + 1));
  } catch (...) {
    *error = "SymmetricPrincipalSqrtXlaImpl: cannot parse opaque shape";
    return false;
  }
  return *batch > 0 && *n > 0;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void SymmetricSylvesterXlaImpl(
    void* out_raw, const void** in, const char* opaque, size_t opaque_len,
    XlaCustomCallStatus* status) {
  int batch = 0;
  int parameters = 0;
  int n = 0;
  std::string error;
  if (!parse_opaque(opaque, opaque_len, &batch, &parameters, &n, &error)) {
    XlaCustomCallStatusSetFailure(status, error.c_str(), error.size());
    return;
  }

  const double* s = static_cast<const double*>(in[0]);
  const double* rhs = static_cast<const double*>(in[1]);
  void** out = static_cast<void**>(out_raw);
  double* x = static_cast<double*>(out[0]);

  const int info = solve_batch(batch, parameters, n, s, rhs, x, &error);
  if (info != 0) {
    std::memset(
        x, 0,
        static_cast<size_t>(batch) * parameters * n * n * sizeof(double));
    XlaCustomCallStatusSetFailure(status, error.c_str(), error.size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "SymmetricSylvesterXlaImpl", SymmetricSylvesterXlaImpl, "Host");

extern "C" __attribute__((visibility("default"))) void
SymmetricPrincipalSqrtXlaImpl(void* out_raw, const void** in,
                              const char* opaque, size_t opaque_len,
                              XlaCustomCallStatus* status) {
  int batch = 0;
  int n = 0;
  std::string error;
  if (!parse_matrix_opaque(opaque, opaque_len, &batch, &n, &error)) {
    XlaCustomCallStatusSetFailure(status, error.c_str(), error.size());
    return;
  }

  const double* covariance = static_cast<const double*>(in[0]);
  void** out = static_cast<void**>(out_raw);
  double* factor = static_cast<double*>(out[0]);

  const int info =
      principal_sqrt_batch(batch, n, covariance, factor, &error);
  if (info != 0) {
    std::memset(factor, 0,
                static_cast<size_t>(batch) * n * n * sizeof(double));
    XlaCustomCallStatusSetFailure(status, error.c_str(), error.size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "SymmetricPrincipalSqrtXlaImpl", SymmetricPrincipalSqrtXlaImpl, "Host");

#ifdef HAS_CUDA

extern "C" __attribute__((visibility("default"))) void
SymmetricSylvesterXlaImpl_Gpu(cudaStream_t stream, void** buffers,
                              const char* opaque, size_t opaque_len,
                              XlaCustomCallStatus* status) {
  int batch = 0;
  int parameters = 0;
  int n = 0;
  std::string error;
  if (!parse_opaque(opaque, opaque_len, &batch, &parameters, &n, &error)) {
    const std::string message =
        "SymmetricSylvesterXlaImpl (GPU): " + error;
    XlaCustomCallStatusSetFailure(status, message.c_str(), message.size());
    return;
  }

  const size_t nn = static_cast<size_t>(n) * n;
  const size_t s_bytes = static_cast<size_t>(batch) * nn * sizeof(double);
  const size_t rhs_bytes =
      static_cast<size_t>(batch) * parameters * nn * sizeof(double);

  const void* d_s = buffers[0];
  const void* d_rhs = buffers[1];
  void* d_x = buffers[2];

  std::vector<double> h_s(static_cast<size_t>(batch) * nn);
  std::vector<double> h_rhs(static_cast<size_t>(batch) * parameters * nn);
  std::vector<double> h_x(static_cast<size_t>(batch) * parameters * nn);

  cudaDeviceSynchronize();
  cudaMemcpyAsync(h_s.data(), d_s, s_bytes, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(
      h_rhs.data(), d_rhs, rhs_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  const int info = solve_batch(
      batch, parameters, n, h_s.data(), h_rhs.data(), h_x.data(), &error);
  if (info != 0) {
    std::memset(h_x.data(), 0, rhs_bytes);
    XlaCustomCallStatusSetFailure(status, error.c_str(), error.size());
  }

  cudaMemcpyAsync(d_x, h_x.data(), rhs_bytes, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "SymmetricSylvesterXlaImpl", SymmetricSylvesterXlaImpl_Gpu, "CUDA");

extern "C" __attribute__((visibility("default"))) void
SymmetricPrincipalSqrtXlaImpl_Gpu(cudaStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len,
                                  XlaCustomCallStatus* status) {
  int batch = 0;
  int n = 0;
  std::string error;
  if (!parse_matrix_opaque(opaque, opaque_len, &batch, &n, &error)) {
    const std::string message =
        "SymmetricPrincipalSqrtXlaImpl (GPU): " + error;
    XlaCustomCallStatusSetFailure(status, message.c_str(), message.size());
    return;
  }

  const size_t nn = static_cast<size_t>(n) * n;
  const size_t bytes = static_cast<size_t>(batch) * nn * sizeof(double);

  const void* d_covariance = buffers[0];
  void* d_factor = buffers[1];

  std::vector<double> h_covariance(static_cast<size_t>(batch) * nn);
  std::vector<double> h_factor(static_cast<size_t>(batch) * nn);

  cudaDeviceSynchronize();
  cudaMemcpyAsync(
      h_covariance.data(), d_covariance, bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  const int info = principal_sqrt_batch(
      batch, n, h_covariance.data(), h_factor.data(), &error);
  if (info != 0) {
    std::memset(h_factor.data(), 0, bytes);
    XlaCustomCallStatusSetFailure(status, error.c_str(), error.size());
  }

  cudaMemcpyAsync(d_factor, h_factor.data(), bytes, cudaMemcpyHostToDevice,
                  stream);
  cudaStreamSynchronize(stream);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "SymmetricPrincipalSqrtXlaImpl", SymmetricPrincipalSqrtXlaImpl_Gpu,
    "CUDA");

#endif  // HAS_CUDA

namespace tensorflow {
namespace {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SymmetricSylvester")
    .Input("s: double")
    .Input("rhs: double")
    .Output("x: double")
    .SetShapeFn([](InferenceContext* c) -> Status {
      ShapeHandle s_shape;
      ShapeHandle rhs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &s_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &rhs_shape));
      c->set_output(0, rhs_shape);
      return OkStatus();
    });

class SymmetricSylvesterOp : public OpKernel {
 public:
  explicit SymmetricSylvesterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& s_t = ctx->input(0);
    const Tensor& rhs_t = ctx->input(1);
    OP_REQUIRES(ctx, s_t.dims() == 3,
                errors::InvalidArgument("s must have shape [batch,n,n]"));
    OP_REQUIRES(ctx, rhs_t.dims() == 4,
                errors::InvalidArgument("rhs must have shape [batch,p,n,n]"));
    const int64_t batch = s_t.dim_size(0);
    const int64_t n = s_t.dim_size(1);
    const int64_t parameters = rhs_t.dim_size(1);
    OP_REQUIRES(ctx, n > 0 && s_t.dim_size(2) == n,
                errors::InvalidArgument("s must be square on trailing axes"));
    OP_REQUIRES(ctx, rhs_t.dim_size(0) == batch &&
                         rhs_t.dim_size(2) == n && rhs_t.dim_size(3) == n,
                errors::InvalidArgument(
                    "rhs must have shape [batch,p,n,n] matching s"));

    Tensor* x_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, rhs_t.shape(), &x_t));

    std::string error;
    const int info = solve_batch(
        static_cast<int>(batch), static_cast<int>(parameters),
        static_cast<int>(n), s_t.flat<double>().data(),
        rhs_t.flat<double>().data(), x_t->flat<double>().data(), &error);
    OP_REQUIRES(ctx, info == 0, errors::InvalidArgument(error));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SymmetricSylvester").Device(DEVICE_CPU), SymmetricSylvesterOp);

REGISTER_OP("SymmetricPrincipalSqrt")
    .Input("covariance: double")
    .Output("factor: double")
    .SetShapeFn([](InferenceContext* c) -> Status {
      ShapeHandle covariance_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &covariance_shape));
      c->set_output(0, covariance_shape);
      return OkStatus();
    });

class SymmetricPrincipalSqrtOp : public OpKernel {
 public:
  explicit SymmetricPrincipalSqrtOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& covariance_t = ctx->input(0);
    OP_REQUIRES(ctx, covariance_t.dims() == 3,
                errors::InvalidArgument(
                    "covariance must have shape [batch,n,n]"));
    const int64_t batch = covariance_t.dim_size(0);
    const int64_t n = covariance_t.dim_size(1);
    OP_REQUIRES(ctx, batch > 0 && n > 0 && covariance_t.dim_size(2) == n,
                errors::InvalidArgument(
                    "covariance must be square on trailing axes"));

    Tensor* factor_t = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, covariance_t.shape(), &factor_t));

    std::string error;
    const int info = principal_sqrt_batch(
        static_cast<int>(batch), static_cast<int>(n),
        covariance_t.flat<double>().data(), factor_t->flat<double>().data(),
        &error);
    OP_REQUIRES(ctx, info == 0, errors::InvalidArgument(error));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SymmetricPrincipalSqrt").Device(DEVICE_CPU),
    SymmetricPrincipalSqrtOp);

class SymmetricSylvesterXlaOp : public XlaOpKernel {
 public:
  explicit SymmetricSylvesterXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape s_shape = ctx->InputShape(0);
    const TensorShape rhs_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, s_shape.dims() == 3,
                errors::InvalidArgument("s must have rank 3"));
    OP_REQUIRES(ctx, rhs_shape.dims() == 4,
                errors::InvalidArgument("rhs must have rank 4"));
    const int64_t batch = s_shape.dim_size(0);
    const int64_t n = s_shape.dim_size(1);
    const int64_t parameters = rhs_shape.dim_size(1);
    OP_REQUIRES(ctx, batch > 0 && parameters > 0 && n > 0,
                errors::InvalidArgument("batch, parameter, and n must be > 0"));
    OP_REQUIRES(ctx, s_shape.dim_size(2) == n &&
                         rhs_shape.dim_size(0) == batch &&
                         rhs_shape.dim_size(2) == n &&
                         rhs_shape.dim_size(3) == n,
                errors::InvalidArgument(
                    "rhs must have shape [batch,p,n,n] matching s"));

    // The host/CUDA callbacks index raw buffers in C row-major order. Pin all
    // leaf layouts so XLA cannot choose a different physical layout inside a
    // larger while-loop computation. Batch size one can mask this mismatch.
    const xla::Shape s_layout = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::F64, {batch, n, n}, {2, 1, 0});
    const xla::Shape rhs_layout = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::F64, {batch, parameters, n, n}, {3, 2, 1, 0});
    const xla::Shape out_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::F64, {batch, parameters, n, n}, {3, 2, 1, 0});
    const xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape({out_shape});
    const std::vector<xla::Shape> operand_layouts = {s_layout, rhs_layout};
    const std::string opaque = std::to_string(batch) + "," +
                               std::to_string(parameters) + "," +
                               std::to_string(n);
    const xla::XlaOp result = xla::CustomCallWithLayout(
        ctx->builder(), "SymmetricSylvesterXlaImpl",
        {ctx->Input(0), ctx->Input(1)}, tuple_shape, operand_layouts, opaque,
        /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        xla::CustomCallSchedule::SCHEDULE_NONE,
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED);
    ctx->SetOutput(0, xla::GetTupleElement(result, 0));
  }
};

REGISTER_XLA_OP(Name("SymmetricSylvester"), SymmetricSylvesterXlaOp);

class SymmetricPrincipalSqrtXlaOp : public XlaOpKernel {
 public:
  explicit SymmetricPrincipalSqrtXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape covariance_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, covariance_shape.dims() == 3,
                errors::InvalidArgument("covariance must have rank 3"));
    const int64_t batch = covariance_shape.dim_size(0);
    const int64_t n = covariance_shape.dim_size(1);
    OP_REQUIRES(ctx, batch > 0 && n > 0 && covariance_shape.dim_size(2) == n,
                errors::InvalidArgument(
                    "covariance must have shape [batch,n,n]"));

    const xla::Shape covariance_layout =
        xla::ShapeUtil::MakeShapeWithDenseLayout(
            xla::F64, {batch, n, n}, {2, 1, 0});
    const xla::Shape out_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::F64, {batch, n, n}, {2, 1, 0});
    const xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape({out_shape});
    const std::vector<xla::Shape> operand_layouts = {covariance_layout};
    const std::string opaque =
        std::to_string(batch) + "," + std::to_string(n);
    const xla::XlaOp result = xla::CustomCallWithLayout(
        ctx->builder(), "SymmetricPrincipalSqrtXlaImpl", {ctx->Input(0)},
        tuple_shape, operand_layouts, opaque,
        /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        xla::CustomCallSchedule::SCHEDULE_NONE,
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED);
    ctx->SetOutput(0, xla::GetTupleElement(result, 0));
  }
};

REGISTER_XLA_OP(Name("SymmetricPrincipalSqrt"), SymmetricPrincipalSqrtXlaOp);

}  // namespace
}  // namespace tensorflow
