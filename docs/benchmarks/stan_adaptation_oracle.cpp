// Independent diagnostic oracle using the installed, unmodified Stan headers.
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/var_adaptation.hpp>
#include <stan/mcmc/covar_adaptation.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <iostream>
#include <iomanip>
#include <random>
class schedule : public stan::mcmc::windowed_adaptation {
 public:
  schedule():windowed_adaptation("test") {}
  void advance() { if(end_adaptation_window()) compute_next_window(); ++adapt_window_counter_; }
};
// Exercise actual base_hmc::init_stepsize with scripted Hamiltonian differences.
// Only the probe is controlled: the search control flow is unmodified Stan.
struct probe_model {
  std::vector<double> log_accept;
  mutable std::vector<double> epsilons;
  int num_params_r() const { return 2; }
};
template <class Model, class RNG>
class probe_hamiltonian {
 public:
  using PointType = stan::mcmc::diag_e_point;
  const Model& model;
  explicit probe_hamiltonian(const Model& m): model(m) {}
  void sample_p(PointType& z, RNG&) { z.p.setZero(); }
  void init(PointType& z, stan::callbacks::logger&) { z.V=0; }
  double H(PointType& z) { return z.V; }
};
template <class Hamiltonian>
class probe_integrator {
 public:
  void evolve(typename Hamiltonian::PointType& z, Hamiltonian& h, double e,
              stan::callbacks::logger&) {
    size_t i=h.model.epsilons.size();
    h.model.epsilons.push_back(e);
    z.V=-h.model.log_accept[std::min(i,h.model.log_accept.size()-1)];
  }
};
using base_probe = stan::mcmc::base_hmc<
    probe_model,probe_hamiltonian,probe_integrator,std::mt19937>;
class probe_sampler: public base_probe {
 public:
  probe_sampler(const probe_model& m,std::mt19937& r):base_probe(m,r) {}
  stan::mcmc::sample transition(stan::mcmc::sample& s,
                              stan::callbacks::logger&) { return s; }
};

class gaussian_hamiltonian {
 public:
  using PointType = stan::mcmc::dense_e_point;
  Eigen::VectorXd dphi_dq(PointType& z,stan::callbacks::logger&) { return z.g; }
  Eigen::VectorXd dtau_dp(PointType& z) { return z.inv_e_metric_*z.p; }
  void update_potential_gradient(PointType& z,stan::callbacks::logger&) {
    z.g=z.q; z.g[1]*=.25; z.V=.5*z.q.dot(z.g);
  }
  double H(PointType& z) { return z.V+.5*z.p.dot(z.inv_e_metric_*z.p); }
};

int main(int argc, char** argv) {
  std::cout << std::setprecision(17);
  stan::callbacks::logger logger;
  if(argc>1 && std::string(argv[1])=="leapfrog") {
    stan::mcmc::dense_e_point z(2);
    z.inv_e_metric_ << 4.,2.,2.,10.;
    z.q << .8,-.2;
    z.p << 0.,.1;
    gaussian_hamiltonian h;
    h.update_potential_gradient(z,logger);
    double initial=h.H(z);
    stan::mcmc::expl_leapfrog<gaussian_hamiltonian> integrator;
    integrator.evolve(z,h,std::stod(argv[2]),logger);
    std::cout << z.q[0] << " " << z.q[1] << " " << z.p[0] << " "
              << z.p[1] << " " << initial-h.H(z) << std::endl;
    return 0;
  }
  if(argc>1 && std::string(argv[1])=="search") {
    probe_model m;
    for(int i=3;i<argc;++i) m.log_accept.push_back(std::stod(argv[i]));
    std::mt19937 rng(17);
    probe_sampler sampler(m,rng);
    sampler.z().q << .3,-.4;
    const Eigen::VectorXd initial=sampler.z().q;
    sampler.set_nominal_stepsize(std::stod(argv[2]));
    bool success=true;
    try { sampler.init_stepsize(logger); }
    catch(const std::runtime_error&) { success=false; }
    std::cout << success << " " << sampler.get_nominal_stepsize()
              << " " << m.epsilons.size()
              << " " << (sampler.z().q-initial).norm();
    for(double e:m.epsilons) std::cout << " " << e;
    std::cout << std::endl;
    return 0;
  }
  if(argc>1 && std::string(argv[1])=="schedule") {
    int n=std::stoi(argv[2]);
    schedule w; w.set_window_params(n,std::stoi(argv[3]),std::stoi(argv[4]),
                                   std::stoi(argv[5]),logger);
    for(int i=0;i<n;++i) {
      std::cout << i << " " << w.adaptation_window()
                << " " << w.end_adaptation_window() << std::endl;
      w.advance();
    }
    return 0;
  }
  if(argc>1 && std::string(argv[1])=="replay") {
    // Source-ordered composition on supplied positions/acceptances. Boundary
    // epsilon is supplied; its search is independently tested above.
    int n=std::stoi(argv[2]), initial=std::stoi(argv[3]);
    int terminal=std::stoi(argv[4]), window=std::stoi(argv[5]);
    bool dense=std::string(argv[6])=="dense";
    stan::mcmc::stepsize_adaptation da;
    da.set_delta(std::stod(argv[7])); da.set_gamma(std::stod(argv[8]));
    da.set_kappa(std::stod(argv[9])); da.set_t0(std::stod(argv[10]));
    double e=std::stod(argv[11]); da.set_mu(log(10*e));
    stan::mcmc::covar_adaptation cov(2); stan::mcmc::var_adaptation var(2);
    cov.set_window_params(n,initial,terminal,window,logger);
    var.set_window_params(n,initial,terminal,window,logger);
    Eigen::MatrixXd c=Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd v=Eigen::VectorXd::Ones(2), q(2);
    for(int i=0;i<n;++i) {
      double accept,searched;
      if(!(std::cin>>q[0]>>q[1]>>accept>>searched)) return 2;
      da.learn_stepsize(e,accept);
      bool update=dense?cov.learn_covariance(c,q):var.learn_variance(v,q);
      if(update) { e=searched; da.set_mu(log(10*e)); da.restart(); }
      if(!dense) c=v.asDiagonal();
      std::cout << i << " " << e << " " << update << " "
                << c(0,0) << " " << c(0,1) << " " << c(1,0) << " "
                << c(1,1) << std::endl;
    }
    da.complete_adaptation(e);
    std::cout << "final " << e << std::endl;
    return 0;
  }
  if(argc>1 && std::string(argv[1])=="stream") {
    int n=std::stoi(argv[2]), initial=std::stoi(argv[3]);
    int terminal=std::stoi(argv[4]), window=std::stoi(argv[5]);
    stan::mcmc::covar_adaptation cov(2); stan::mcmc::var_adaptation var(2);
    cov.set_window_params(n,initial,terminal,window,logger);
    var.set_window_params(n,initial,terminal,window,logger);
    Eigen::MatrixXd c=Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd v=Eigen::VectorXd::Ones(2),q(2);
    for(int i=0;i<n;++i) {
      q << .1*i,.01*i*i;
      bool update=cov.learn_covariance(c,q); var.learn_variance(v,q);
      if(update) std::cout << i << " " << c(0,0) << " " << c(0,1) << " "
                          << c(1,0) << " " << c(1,1) << " " << v(0)
                          << " " << v(1) << std::endl;
    }
    return 0;
  }
  for (int n : {150,200,1000}) {
    schedule w; w.set_window_params(n,75,50,25,logger);
    std::cout << "schedule " << n;
    for(int i=0;i<n;++i) { if(w.end_adaptation_window()) std::cout << " " << i; w.advance(); }
    std::cout << std::endl;
  }
  stan::mcmc::stepsize_adaptation da;
  da.set_delta(.8); da.set_mu(log(10.));
  double e=1;
  for(int i=0;i<12;++i) {
    if(i==6) { da.set_mu(log(2.)); da.restart(); }
    da.learn_stepsize(e,.15+.07*i);
    std::cout << "da " << i << " " << e << std::endl;
  }
  da.complete_adaptation(e); std::cout << "final " << e << std::endl;
  stan::mcmc::covar_adaptation cov(2);
  stan::mcmc::var_adaptation var(2);
  cov.set_window_params(30,5,5,20,logger);
  var.set_window_params(30,5,5,20,logger);
  Eigen::MatrixXd c(2,2); Eigen::VectorXd v(2), q(2);
  for(int i=0;i<30;++i) {
    q << .1*i, .01*i*i;
    bool update=cov.learn_covariance(c,q); var.learn_variance(v,q);
    if(update) std::cout << "cov " << c(0,0) << " " << c(0,1) << " " << c(1,0) << " " << c(1,1)
                        << " " << v(0) << " " << v(1) << std::endl;
  }
}
