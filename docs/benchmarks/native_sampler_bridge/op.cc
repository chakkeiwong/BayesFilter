// Standalone hybrid XLA bridge: native samplers always execute on CPU.
#include "native.h"
#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"
namespace {
using Config=std::array<int64_t,8>;
Config parse(const char* opaque,size_t size) {
  Config c;std::istringstream in(std::string(opaque,size));
  for(auto& v:c) if(!(in>>v))throw std::invalid_argument("bad native sampler config");
  return c;
}
void failure(XlaCustomCallStatus* s,const std::exception& e) {if(s)s->message=e.what();}
void invoke(const Config& c,const double* controls,const double* starts,const double* inv,
    const int64_t* seeds,double* rows,double* meta) {
  char error[2048]={};
  if(bf_native_fit(c.data(),controls,starts,inv,seeds,rows,meta,error,sizeof(error)))
    throw std::runtime_error(error);
}
void cpu(void* raw,const void** inputs,const char* opaque,size_t size,XlaCustomCallStatus* status) {
  try {auto c=parse(opaque,size);auto out=static_cast<void**>(raw);
    invoke(c,static_cast<const double*>(inputs[0]),static_cast<const double*>(inputs[1]),
      static_cast<const double*>(inputs[2]),static_cast<const int64_t*>(inputs[3]),
      static_cast<double*>(out[0]),static_cast<double*>(out[1]));
  } catch(const std::exception& e) {failure(status,e);}
  catch(...) {if(status)status->message="unknown CPU bridge error";}
}
void cuda_check(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
void gpu(cudaStream_t stream,void** b,const char* opaque,size_t size,XlaCustomCallStatus* status) {
  // Retain staging buffers until all copies finish, including on errors.
  std::vector<double> controls,starts,inv,rows,meta;std::vector<int64_t> seeds;
  try {auto c=parse(opaque,size);const size_t d=c[2],chains=c[3],iters=c[4]+c[5];
    controls.resize(4);starts.resize(chains*d);inv.resize(d*d);seeds.resize(chains);
    rows.resize(chains*iters*(d+7));meta.resize(chains*(d+4));
    cuda_check(cudaMemcpyAsync(controls.data(),b[0],controls.size()*8,cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaMemcpyAsync(starts.data(),b[1],starts.size()*8,cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaMemcpyAsync(inv.data(),b[2],inv.size()*8,cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaMemcpyAsync(seeds.data(),b[3],seeds.size()*8,cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaStreamSynchronize(stream));
    invoke(c,controls.data(),starts.data(),inv.data(),seeds.data(),rows.data(),meta.data());
    cuda_check(cudaMemcpyAsync(b[4],rows.data(),rows.size()*8,cudaMemcpyHostToDevice,stream));
    cuda_check(cudaMemcpyAsync(b[5],meta.data(),meta.size()*8,cudaMemcpyHostToDevice,stream));
    cuda_check(cudaStreamSynchronize(stream));
  } catch(const std::exception& e) {cudaStreamSynchronize(stream);failure(status,e);}
  catch(...) {cudaStreamSynchronize(stream);if(status)status->message="unknown GPU bridge error";}
}
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("BfNativeSamplerV1",cpu,"Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("BfNativeSamplerV1",gpu,"CUDA");
}
namespace tensorflow {
REGISTER_OP("BfNativeSampler")
    .Input("controls: double").Input("starts: double").Input("inverse_transform: double").Input("seeds: int64")
    .Output("rows: double").Output("metadata: double")
    .Attr("engine: int").Attr("kind: int").Attr("warmup: int >= 1").Attr("draws: int >= 1")
    .Attr("depth: int >= 1").Attr("halvings: int >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle start;TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&start));
      int w,s;TF_RETURN_IF_ERROR(c->GetAttr("warmup",&w));TF_RETURN_IF_ERROR(c->GetAttr("draws",&s));
      shape_inference::DimensionHandle width,meta;
      TF_RETURN_IF_ERROR(c->Add(c->Dim(start,1),7,&width));
      TF_RETURN_IF_ERROR(c->Add(c->Dim(start,1),4,&meta));
      c->set_output(0,c->MakeShape({c->Dim(start,0),c->MakeDim(int64_t(w)+s),width}));
      c->set_output(1,c->MakeShape({c->Dim(start,0),meta}));return absl::OkStatus();
    });
class NativeSamplerOp : public XlaOpKernel {
 public:
  explicit NativeSamplerOp(OpKernelConstruction* c):XlaOpKernel(c) {
    OP_REQUIRES_OK(c,c->GetAttr("engine",&config_[0]));OP_REQUIRES_OK(c,c->GetAttr("kind",&config_[1]));
    OP_REQUIRES_OK(c,c->GetAttr("warmup",&config_[4]));OP_REQUIRES_OK(c,c->GetAttr("draws",&config_[5]));
    OP_REQUIRES_OK(c,c->GetAttr("depth",&config_[6]));OP_REQUIRES_OK(c,c->GetAttr("halvings",&config_[7]));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto starts=ctx->InputShape(1);
    OP_REQUIRES(ctx,starts.dims()==2,errors::InvalidArgument("starts must be rank two"));
    config_[2]=starts.dim_size(1);config_[3]=starts.dim_size(0);
    OP_REQUIRES(ctx,config_[4]<=10000000&&config_[5]<=10000000,
      errors::InvalidArgument("iteration counts exceed bridge bound"));
    auto d=config_[2],chains=config_[3],iters=config_[4]+config_[5];
    OP_REQUIRES(ctx,d>0&&d<=4096&&chains>0&&chains<=64&&iters<=10000000&&
      chains*iters*(d+7)<=100000000&&config_[6]<=15&&config_[7]<=10,
      errors::InvalidArgument("invalid/bounded native sampler shape or configuration"));
    OP_REQUIRES(ctx,ctx->InputShape(0)==TensorShape({4})&&ctx->InputShape(2)==TensorShape({d,d})&&
      ctx->InputShape(3)==TensorShape({chains}),errors::InvalidArgument("invalid controls, matrix or seed shape"));
    auto shape=[](xla::PrimitiveType type,std::vector<int64_t> dims) {
      std::vector<int64_t> layout;for(int i=int(dims.size())-1;i>=0;--i)layout.push_back(i);
      return xla::ShapeUtil::MakeShapeWithDenseLayout(type,dims,layout);};
    std::vector<xla::Shape> inputs={shape(xla::F64,{4}),shape(xla::F64,{chains,d}),
      shape(xla::F64,{d,d}),shape(xla::S64,{chains})};
    auto outputs=xla::ShapeUtil::MakeTupleShape({shape(xla::F64,{chains,iters,d+7}),shape(xla::F64,{chains,d+4})});
    std::ostringstream opaque;for(auto v:config_)opaque<<v<<' ';
    auto result=xla::CustomCallWithLayout(ctx->builder(),"BfNativeSamplerV1",
      {ctx->Input(0),ctx->Input(1),ctx->Input(2),ctx->Input(3)},outputs,inputs,opaque.str(),
      false,{},nullptr,xla::CustomCallSchedule::SCHEDULE_NONE,
      xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED);
    ctx->SetOutput(0,xla::GetTupleElement(result,0));ctx->SetOutput(1,xla::GetTupleElement(result,1));
  }
 private:Config config_{};
};
REGISTER_XLA_OP(Name("BfNativeSampler"),NativeSamplerOp);
}
