// Independent reference: unmodified pinned Walnutpie headers, MIT (see LICENSE).
// Tape specialization supplies randomness only; numerical algorithms are upstream.
#include <walnutpie/util.hpp>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
struct Tape {
 using result_type=uint64_t; int key=1,counter=0;
 static constexpr result_type min(){return 0;}
 static constexpr result_type max(){return UINT64_MAX;}
 result_type operator()(){return 0;}
};
namespace walnutpie::detail {
template<> class Random<Tape> {
 Tape& r_;
 public:
 explicit Random(Tape& r):r_(r){}
 double uniform_real_01(){double u=((r_.counter*73+r_.key*19)%997+.5)/997.;++r_.counter;return u;}
 bool uniform_binary(){return uniform_real_01()<.5;}
 Eigen::VectorXd standard_normal(Eigen::Index d){
  Eigen::VectorXd z(d);for(int j=0;j<d;++j)z[j]=((r_.key*13+j*17)%101-50)/19.;++r_.counter;return z;
 }
 Tape& rng(){return r_;}
};
}
#include <walnutpie/adaptive_walnuts.hpp>
using Vec=Eigen::VectorXd;
struct Target {
 int kind,d; mutable long evaluations=0; Eigen::MatrixXd x; Vec y;
 void operator()(const Vec& q,double& lp,Vec& g) const {
  ++evaluations;g.resize(d);
  if(kind==0){lp=-.5*q.squaredNorm();g=-q;}
  else if(kind==1){lp=0;g.setZero();for(int j=0;j<d;j+=2){double z=q[j+1]-.3*(q[j]*q[j]-1);lp-=.5*(q[j]*q[j]+z*z);g[j]=-q[j]+.6*q[j]*z;g[j+1]=-z;}}
  else if(kind==2){double s=std::exp(-q[0]),ss=q.tail(d-1).squaredNorm();lp=-q[0]*q[0]/18.-.5*(d-1)*q[0]-.5*s*ss;g[0]=-q[0]/9.-.5*(d-1)+.5*s*ss;g.tail(d-1)=-s*q.tail(d-1);}
  else {Vec r=x*q-y;lp=-.5*q.squaredNorm()-.5*r.squaredNorm()/x.rows();g=-q-x.transpose()*r/x.rows();}
 }
};
void vec(const Vec& v){std::cout<<'[';for(int j=0;j<v.size();++j){if(j)std::cout<<',';if(std::isfinite(v[j]))std::cout<<v[j];else std::cout<<"null";}std::cout<<']';}
struct Observe {
 walnutpie::detail::Adam a; int count=0;
 explicit Observe(double e):a(e,.8,.05,.8,.9,1e-4,.5){}
 void operator()(double alpha){++count;a(alpha);}
 double step_size()const{return a.step_size();}
};
struct Handler {
 Vec q;double lp=0;std::vector<Vec> draws;
 void on_warmup(const Vec& x,double l,double,const Vec&){q=x;lp=l;}
 void on_warmup_complete(double,const Vec&){}
 void on_sample(const Vec& x,double l){q=x;lp=l;draws.push_back(x);}
 void on_logp_exception(const Vec&,const std::exception&)noexcept{}
};
void oracle(){
 using namespace walnutpie::detail;
 for(int kind=0;kind<4;++kind)for(int key=1;key<=24;++key){
  int d=kind==1?4:5;Target f{kind,d};Vec q(d),g,inv(d);
  if(kind==3){f.x.resize(8,d);f.y.resize(8);for(int i=0;i<8;++i){f.y[i]=.1*(i-3);for(int j=0;j<d;++j)f.x(i,j)=.1*((i*7+j*3)%11-5);}}
  for(int j=0;j<d;++j){q[j]=.15*((key+j*3)%11-5);inv[j]=.5+.1*((key+j)%9);}
  if(kind==2)q[0]=(key%9)-4.;
  double lp;f(q,lp,g); double epsilon=std::pow(2.,key%7-3);int minimum=1+key%3;
  auto header=[&](const char* type){std::cout<<"{\"type\":\""<<type<<"\",\"kind\":"<<kind<<",\"key\":"<<key<<",\"q\":";vec(q);std::cout<<",\"inverse_mass\":";vec(inv);std::cout<<",\"epsilon\":"<<epsilon<<",\"minimum\":"<<minimum;};
  {Tape rng{key};Random<Tape> rand(rng);Observe a(epsilon);size_t depth;Vec gg=g;double l=lp;f.evaluations=0;
   Vec out=transition_w(rand,f,inv,inv.array().sqrt().inverse().matrix(),epsilon,5,5,minimum,.5,Vec(q),depth,gg,l,a);
   header("transition");std::cout<<",\"out\":";vec(out);std::cout<<",\"gradient\":";vec(gg);
   std::cout<<",\"lp\":"<<l<<",\"depth\":"<<depth<<",\"evaluations\":"<<f.evaluations<<",\"macros\":"<<a.count<<",\"counter\":"<<rng.counter<<",\"adapted_epsilon\":"<<a.step_size()<<"}\n";}
  {Tape rng{key};Random<Tape> rand(rng);Vec p=(rand.standard_normal(d).array()/inv.array().sqrt()).matrix();
   double j=lp+logp_momentum(p,inv);auto span=SpanW::from_initial_point(Vec(q),Vec(p),Vec(g),lp,j);
   Vec oq,op,og;double ol=0,oj=0;Observe a(epsilon);f.evaluations=0;
   bool ok=macro_step<Direction::Forward>(f,inv,epsilon,5,minimum,.5,span,oq,op,og,ol,oj,a);
   header("macro");std::cout<<",\"momentum\":";vec(p);std::cout<<",\"ok\":"<<(ok?"true":"false")<<",\"out\":";vec(oq);
   std::cout<<",\"evaluations\":"<<f.evaluations<<",\"adapted_epsilon\":"<<a.step_size()<<"}\n";}
  {Tape rng{key};Vec mass=(.99999*g.array().abs()+.00001).matrix();
   double e=adapt_step(rng,f,q,mass,.1,d);header("initialize");std::cout<<",\"adapted_epsilon\":"<<e<<"}\n";}
 }
 for(int kind=0;kind<3;++kind)for(double target: {15.,1.}){int d=4;Target f{kind,d};Vec q=Vec::LinSpaced(d,-.4,.5),g;double lp;f(q,lp,g);
  Vec mass=(.99999*g.array().abs()+.00001).matrix();Tape rng{7};double e=adapt_step(rng,f,q,mass,.1,d);
  auto w=walnutpie::WarmupConfigBuilder().max_macro_steps_target(target).build();auto s=walnutpie::SamplingConfigBuilder().build();
  Handler h;walnutpie::InitChainConfig init(e,q,mass);walnutpie::AdaptiveWalnuts a(rng,h,f,init,w,s);
  auto emit=[&](int i,bool warm){std::cout<<"{\"type\":\"lifecycle\",\"kind\":"<<kind<<",\"target_macro_steps\":"<<target<<",\"iteration\":"<<i<<",\"warmup\":"<<(warm?"true":"false")<<",\"q\":";vec(h.q);std::cout<<",\"epsilon\":"<<a.step_size()<<",\"inverse_mass\":";vec(a.inv_mass());std::cout<<",\"minimum\":"<<a.min_micro_steps()<<",\"counter\":"<<rng.counter<<"}\n";};
  for(int i=0;i<30;++i){a();emit(i,true);}auto sample=a.sampler();for(int i=0;i<10;++i){sample();emit(i,false);}
 }
}
struct Result {double seconds;long evaluations;std::vector<Vec> draws;double epsilon;Vec inv;};
Result native_fit(Target f,const Vec& q,int chain,int warm,int draws,int depth){
 auto begin=std::chrono::steady_clock::now();std::mt19937_64 rng(20261028+chain);
 Vec g;double lp;f(q,lp,g);Vec mass=(.99999*g.array().abs()+.00001).matrix();
 double e=walnutpie::detail::adapt_step(rng,f,q,mass,.1,q.size());
 Handler h;auto w=walnutpie::WarmupConfigBuilder().build();auto s=walnutpie::SamplingConfigBuilder().max_trajectory_doublings(depth).build();
 walnutpie::InitChainConfig init(e,q,mass);walnutpie::AdaptiveWalnuts a(rng,h,f,init,w,s);
 for(int i=0;i<warm;++i)a();auto sampler=a.sampler();for(int i=0;i<draws;++i)sampler();
 return {std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(),f.evaluations,h.draws,a.step_size(),a.inv_mass()};
}
void benchmark(const char* file){
 std::ifstream in(file);int kind,d,n,c,w,s,depth;in>>kind>>d>>n>>c>>w>>s>>depth;
 Target f{kind,d};std::vector<Vec> starts(c,Vec(d));for(auto& q:starts)for(int j=0;j<d;++j)in>>q[j];
 f.x.resize(n,d);f.y.resize(n);for(int i=0;i<n;++i)for(int j=0;j<d;++j)in>>f.x(i,j);for(int i=0;i<n;++i)in>>f.y[i];
 if(!in)throw std::runtime_error("bad fixture file");
 std::vector<Result> result(c);std::vector<double> serial,parallel;
 for(int rep=0;rep<7;++rep)for(int arm=0;arm<2;++arm){bool threads=(rep+arm)%2;auto begin=std::chrono::steady_clock::now();
  if(threads){std::vector<std::thread> pool;for(int j=0;j<c;++j)pool.emplace_back([&,j]{result[j]=native_fit(f,starts[j],j,w,s,depth);});for(auto& t:pool)t.join();}
  else for(int j=0;j<c;++j)result[j]=native_fit(f,starts[j],j,w,s,depth);
  (threads?parallel:serial).push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());}
 std::cout<<"{\"serial_seconds\":[";for(int i=0;i<7;++i){if(i)std::cout<<',';std::cout<<serial[i];}std::cout<<"],\"parallel_seconds\":[";for(int i=0;i<7;++i){if(i)std::cout<<',';std::cout<<parallel[i];}
 std::cout<<"],\"chains\":[";for(int j=0;j<c;++j){if(j)std::cout<<',';std::cout<<"{\"evaluations\":"<<result[j].evaluations<<",\"epsilon\":"<<result[j].epsilon<<",\"draws\":[";for(int i=0;i<s;++i){if(i)std::cout<<',';vec(result[j].draws[i]);}std::cout<<"]}";}std::cout<<"]}\n";
}
int main(int argc,char**argv){std::cout<<std::setprecision(17);if(argc==2)benchmark(argv[1]);else oracle();}
