// Independent diagnostic: unmodified Stan base_nuts and expl_leapfrog.
// Only the target, momentum draw and uniform source are test-controlled.
#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>

struct tape_rng {
  using result_type = double;
  std::vector<double> tape;
  size_t used = 0;
  static double min() { return 0.; }
  static double max() { return 1.; }
  double operator()() {
    if (used >= tape.size()) throw std::runtime_error("Uniform tape exhausted");
    return tape[used++];
  }
};
struct target {
  int kind;
  Eigen::VectorXd momentum;
  int num_params_r() const { return momentum.size(); }
};
template <class Model, class RNG> class analytic_hamiltonian {
 public:
  using PointType = stan::mcmc::dense_e_point;
  const Model& model;
  explicit analytic_hamiltonian(const Model& m):model(m) {}
  void sample_p(PointType& z, RNG&) { z.p=model.momentum; }
  void init(PointType& z, stan::callbacks::logger& l) { update_potential_gradient(z,l); }
  Eigen::VectorXd dtau_dp(PointType& z) { return z.inv_e_metric_*z.p; }
  Eigen::VectorXd dphi_dq(PointType& z,stan::callbacks::logger&) { return z.g; }
  double H(PointType& z) { return z.V+.5*z.p.dot(z.inv_e_metric_*z.p); }
  void update_potential_gradient(PointType& z,stan::callbacks::logger&) {
    if(model.kind==0) { z.V=.5*z.q.squaredNorm(); z.g=z.q; }
    else if(model.kind==1) {
      double y=z.q[1]-(z.q[0]*z.q[0]-1.);
      z.V=.5*(z.q[0]*z.q[0]+y*y);
      z.g[0]=z.q[0]-2.*z.q[0]*y; z.g[1]=y;
    } else if(model.kind==2) {
      double v=z.q[0],x=z.q[1],r=std::exp(-v);
      z.V=.5*v*v/9.+.5*v+.5*x*x*r;
      z.g[0]=v/9.+.5-.5*x*x*r; z.g[1]=x*r;
    } else if(model.kind==3) {  // bounded domain, invalid proposals become +inf
      z.V=std::abs(z.q[0])<1.? .5*z.q.squaredNorm() : INFINITY;
      z.g=z.q;
    } else { z.V=0.; z.g.setZero(); }  // constant target for exact stopping ties
  }
};
using parent=stan::mcmc::base_nuts<target,analytic_hamiltonian,stan::mcmc::expl_leapfrog,tape_rng>;
class observed_nuts:public parent {
 public:
  std::vector<std::vector<double>> checks;
  observed_nuts(const target& m,tape_rng& r):parent(m,r) {}
  bool compute_criterion(Eigen::VectorXd& b,Eigen::VectorXd& e,Eigen::VectorXd& rho) override {
    std::vector<double> row;
    for(auto* v:{&b,&e,&rho}) for(int i=0;i<v->size();++i) row.push_back((*v)[i]);
    checks.push_back(row);
    return parent::compute_criterion(b,e,rho);
  }
};
int main() {
  int d,kind,depth,count;
  double epsilon,jitter,max_delta;
  if(!(std::cin>>d>>kind>>depth>>epsilon>>jitter>>max_delta)) return 2;
  target m; m.kind=kind; m.momentum.resize(d);
  Eigen::VectorXd q(d); Eigen::MatrixXd metric(d,d);
  for(int i=0;i<d;++i) std::cin>>q[i];
  for(int i=0;i<d;++i) std::cin>>m.momentum[i];
  for(int i=0;i<d;++i) for(int j=0;j<d;++j) std::cin>>metric(i,j);
  tape_rng rng; std::cin>>count;
  for(int i=0;i<count;++i) { double u;std::cin>>u;rng.tape.push_back(u); }
  observed_nuts sampler(m,rng); sampler.set_metric(metric);
  sampler.set_max_depth(depth);sampler.set_nominal_stepsize(epsilon);
  sampler.set_stepsize_jitter(jitter);sampler.set_max_delta(max_delta);
  stan::callbacks::logger logger;
  stan::mcmc::sample init(q,0.,0.);
  auto out=sampler.transition(init,logger);
  std::cout<<std::setprecision(17);
  std::cout<<out.accept_stat()<<" "<<sampler.n_leapfrog_<<" "<<sampler.depth_
    <<" "<<sampler.divergent_<<" "<<sampler.energy_<<" "<<rng.used
    <<" "<<sampler.get_current_stepsize()<<"\n";
  auto& z=sampler.z();
  for(auto* v:{&z.q,&z.p,&z.g}) { for(int i=0;i<d;++i) std::cout<<(*v)[i]<<" ";std::cout<<"\n"; }
  std::cout<<z.V<<"\n";
  for(const auto& row:sampler.checks) { for(double x:row)std::cout<<x<<" ";std::cout<<"\n"; }
}
