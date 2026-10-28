// Uses upstream implementations unchanged. No Python callback or I/O in fits.
#include "native.h"
#include "native_targets.hpp"
#include <stan/io/array_var_context.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <walnutpie/adaptive_walnuts.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using native_targets = native_targets_namespace::native_targets;
std::mutex fit_mutex;  // Protect Stan autodiff arena across callers.
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
void require(bool b, const char* message) { if (!b) throw std::invalid_argument(message); }
void finite(const double* x, size_t n) {
  for(size_t i=0;i<n;++i) require(std::isfinite(x[i]), "non-finite input");
}
stan::io::array_var_context data_context(int kind, int d, double bend, const double* inverse) {
  require(kind>=0 && kind<=3, "kind must be 0..3");
  require(d>0 && d<=4096, "dimension out of bounds");
  require(kind!=2 || d%2==0, "banana requires even dimension");
  require(kind!=3 || d>=2, "funnel requires at least two dimensions");
  finite(inverse, size_t(d)*d); finite(&bend,1);
  std::vector<double> values; values.reserve(size_t(d)*d+1);
  for(int j=0;j<d;++j) for(int i=0;i<d;++i) values.push_back(inverse[i*d+j]);
  values.push_back(bend);
  return stan::io::array_var_context({"inverse_transform","bend","design"},values,
      {{size_t(d),size_t(d)},{},{0,size_t(d)}},{"d","kind","n"},{d,kind,0},{{},{},{}});
}
struct Logger : stan::callbacks::logger {
  std::string last_error;
  void error(const std::string& s) override { last_error=s; }
  void error(const std::stringstream& s) override { last_error=s.str(); }
};
struct Rows : stan::callbacks::writer {
  double* out; size_t width, capacity, count=0;
  Rows(double* p,size_t w,size_t n):out(p),width(w),capacity(n){}
  void operator()(const std::vector<double>& x) override {
    require(x.size()==width && count<capacity, "Stan sample writer shape/count mismatch");
    std::copy(x.begin(),x.end(),out+(count++)*width);
  }
};
struct Metric : stan::callbacks::structured_writer {
  double* out; int dim; bool got_step=false,got_mass=false;
  Metric(double* p,int d):out(p),dim(d){}
  void write(const std::string& k,double v) override {
    if(k=="stepsize") {out[0]=v;got_step=true;}
  }
  void write(const std::string& k,const Eigen::VectorXd& v) override {
    if(k=="inv_metric") {require(v.size()==dim,"metric dimension mismatch");
      std::copy(v.data(),v.data()+dim,out+1);got_mass=true;}
  }
};
struct WalnutHandler {
  double* rows; double* meta; int dim; size_t capacity,count=0,exceptions=0;
  double epsilon=missing;
  void append(const Eigen::VectorXd& q,double lp,double step) {
    require(count<capacity && q.size()==dim,"Walnutpie output shape/count mismatch");
    double* r=rows+(count++)*(dim+7); r[0]=lp; r[2]=step;
    std::copy(q.data(),q.data()+dim,r+7);
  }
  void on_warmup(const Eigen::VectorXd& q,double lp,double step,const Eigen::VectorXd&) {append(q,lp,step);}
  void on_warmup_complete(double step,const Eigen::VectorXd& mass) {
    epsilon=step;meta[0]=step;std::copy(mass.data(),mass.data()+dim,meta+1);
  }
  void on_sample(const Eigen::VectorXd& q,double lp) {append(q,lp,epsilon);}
  void on_logp_exception(const Eigen::VectorXd&,const std::exception&) noexcept {++exceptions;}
};
}
extern "C" int bf_native_fit(const int64_t* c,const double* t,const double* starts,
    const double* inverse,const int64_t* seeds,double* out,double* meta,char* error,size_t nerr) noexcept {
  try {
    std::lock_guard<std::mutex> lock(fit_mutex);
    require(c&&t&&starts&&inverse&&seeds&&out&&meta,"null buffer");
    require(c[0]==0||c[0]==1,"unknown engine");
    require(c[2]>0&&c[2]<=4096&&c[3]>0&&c[3]<=64,"invalid dimension/chains");
    require(c[1]>=0&&c[1]<=3,"kind must be 0..3");
    require(c[4]>0&&c[5]>0&&c[4]<=10000000&&c[5]<=10000000&&c[4]+c[5]<=10000000,"invalid iteration counts");
    require(c[6]>0&&c[6]<=15&&c[7]>=0&&c[7]<=10,"invalid trajectory bounds");
    require(c[3]*(c[4]+c[5])*(c[2]+7)<=100000000,"output exceeds bounded bridge allocation");
    finite(t,4); require(t[1]>0&&t[2]>0&&t[2]<1&&t[3]>0,"invalid sampler controls");
    int d=c[2],chains=c[3],w=c[4],s=c[5];
    finite(starts,size_t(chains)*d);
    auto data=data_context(c[1],d,t[0],inverse);
    native_targets model(data,0,nullptr);
    std::fill(out,out+size_t(chains)*(w+s)*(d+7),missing);
    std::fill(meta,meta+size_t(chains)*(d+4),missing);
    for(int chain=0;chain<chains;++chain) {
      require(seeds[chain]>=0&&seeds[chain]<=UINT32_MAX,"seed outside uint32 range");
      double* rows=out+size_t(chain)*(w+s)*(d+7);
      double* m=meta+size_t(chain)*(d+4);
      Eigen::VectorXd q=Eigen::Map<const Eigen::VectorXd>(starts+chain*d,d);
      if(c[0]==0) {
        stan::io::array_var_context init({"q"},std::vector<double>(q.data(),q.data()+d),{{size_t(d)}});
        auto inv_metric=stan::services::util::create_unit_e_diag_inv_metric(d);
        stan::callbacks::interrupt interrupt; Logger logger; stan::callbacks::writer noop;
        Rows writer(rows,d+7,w+s); Metric metric(m,d);
        int rc=stan::services::sample::hmc_nuts_diag_e_adapt(model,init,inv_metric,
            // CmdStan still consumes random init-context draws for supplied q.
            // Radius 2 preserves that RNG sequence; q itself is fully supplied.
            seeds[chain],chain+1,2,w,s,1,true,0,t[1],0,c[6],t[2],.05,.75,10,
            75,50,25,interrupt,logger,noop,writer,noop,metric);
        require(rc==0,logger.last_error.empty()?"Stan service failed":logger.last_error.c_str());
        require(writer.count==size_t(w+s)&&metric.got_step&&metric.got_mass,"incomplete Stan result");
      } else {
        WalnutHandler handler{rows,m,d,size_t(w+s)};
        size_t gradient_calls=0;
        auto target=[&](const Eigen::VectorXd& x,double& lp,Eigen::VectorXd& grad) {
          ++gradient_calls; Eigen::VectorXd mutable_x=x;
          lp=stan::model::log_prob_grad<true,true>(model,mutable_x,grad,nullptr);
        };
        double lp; Eigen::VectorXd g;target(q,lp,g);
        require(std::isfinite(lp)&&g.allFinite(),"invalid Walnutpie initial target/gradient");
        auto warm=walnutpie::WarmupConfigBuilder().min_max_iter(w,w).step_accept_rate_target(t[2]).build();
        // Match upstream walnutpy.cpp initialization: gradient-based mass and
        // initial step-size search use a separate RNG from transitions.
        std::seed_seq init_seed{uint32_t(seeds[chain]),2u};std::mt19937_64 init_rng(init_seed);
        auto init=walnutpie::InitConfigBuilder(1,d).positions(q).step_sizes(t[1])
            .masses(target,warm.mass_additive_smoothing()).adapt_step_build(init_rng,target);
        auto sampling=walnutpie::SamplingConfigBuilder().min_max_iter(s,s)
            .max_trajectory_doublings(c[6]).max_step_halvings(c[7]).max_hamiltonian_error(t[3]).build();
        std::seed_seq seed{uint32_t(seeds[chain]),uint32_t(chain+1)};std::mt19937_64 rng(seed);
        walnutpie::AdaptiveWalnuts<decltype(target),std::mt19937_64,WalnutHandler> adapter(rng,handler,target,init.init_chain_config(0),warm,sampling);
        for(int i=0;i<w;++i) adapter();
        m[d+1]=adapter.min_micro_steps();
        auto sampler=adapter.sampler();
        for(int i=0;i<s;++i) sampler();
        require(handler.count==size_t(w+s),"incomplete Walnutpie result");
        m[d+2]=gradient_calls;m[d+3]=handler.exceptions;
      }
      require(std::isfinite(m[0])&&m[0]>0,"invalid adapted step size");
      for(int j=1;j<=d;++j) require(std::isfinite(m[j])&&m[j]>0,"invalid adapted metric");
      for(int i=0;i<w+s;++i) {
        require(std::isfinite(rows[i*(d+7)]),"nonfinite retained log density");
        finite(rows+i*(d+7)+7,d);
      }
    }
    if(error&&nerr) error[0]=0;return 0;
  } catch(const std::exception& e) {if(error&&nerr) std::snprintf(error,nerr,"%s",e.what());return 1;}
  catch(...) {if(error&&nerr) std::snprintf(error,nerr,"unknown native failure");return 2;}
}
extern "C" int bf_native_gradient(int64_t kind,int64_t dim,double bend,const double* inv,
    const double* q,double* result,char* error,size_t nerr) noexcept {
  try {std::lock_guard<std::mutex> lock(fit_mutex);
    require(inv&&q&&result,"null gradient buffer");
    require(kind>=0&&kind<=3&&dim>0&&dim<=4096,"invalid gradient model shape");
    auto data=data_context(kind,dim,bend,inv);native_targets model(data,0,nullptr);
    finite(q,dim);Eigen::VectorXd x=Eigen::Map<const Eigen::VectorXd>(q,dim),g;
    result[0]=stan::model::log_prob_grad<true,true>(model,x,g,nullptr);
    std::copy(g.data(),g.data()+dim,result+1);return 0;
  } catch(const std::exception& e) {if(error&&nerr)std::snprintf(error,nerr,"%s",e.what());return 1;}
  catch(...) {return 2;}
}
