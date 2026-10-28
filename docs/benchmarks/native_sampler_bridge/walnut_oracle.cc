// Independent Gaussian reference through the upstream high-level API.
#include <walnutpie/api.hpp>
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>
struct Handler {
  std::vector<Eigen::VectorXd> q;std::vector<double> lp;double epsilon;Eigen::VectorXd inv;
  void on_warmup(const Eigen::VectorXd& x,double l,double,const Eigen::VectorXd&) {q.push_back(x);lp.push_back(l);}
  void on_warmup_complete(double e,const Eigen::VectorXd& m) {epsilon=e;inv=m;}
  void on_sample(const Eigen::VectorXd& x,double l) {q.push_back(x);lp.push_back(l);}
  void on_logp_exception(const Eigen::VectorXd&,const std::exception&) noexcept {}
};
struct Global {void on_r_hat(double){}};
struct Interrupt {void throw_if_interrupted() const {}};
int main() {
  auto target=[](const Eigen::VectorXd& x,double& lp,Eigen::VectorXd& g){lp=-.5*x.dot(x);g=-x;};
  Eigen::VectorXd q(4);q<<.2,-.4,.6,-.8;
  std::seed_seq ss{101u,2u};std::mt19937_64 rng(ss);
  auto warm=walnutpie::WarmupConfigBuilder().min_max_iter(100,100).build();
  auto init=walnutpie::InitConfigBuilder(1,4).positions(q).step_sizes(.1)
    .masses(target,warm.mass_additive_smoothing()).adapt_step_build(rng,target);
  auto sample=walnutpie::SamplingConfigBuilder().min_max_iter(100,100).build();
  walnutpie::WalnutsConfig config{std::move(init),std::move(warm),std::move(sample)};
  std::vector<Handler> handlers(1);Global global;Interrupt interrupt;
  walnutpie::walnuts<std::mt19937_64>(101,handlers,global,interrupt,target,config);
  auto& h=handlers[0];std::cout<<std::setprecision(17);
  std::cout<<h.q.size()<<' '<<h.epsilon;for(int j=0;j<4;++j)std::cout<<' '<<h.inv[j];std::cout<<'\n';
  for(size_t i=0;i<h.q.size();++i){std::cout<<h.lp[i];for(int j=0;j<4;++j)std::cout<<' '<<h.q[i][j];std::cout<<'\n';}
}
