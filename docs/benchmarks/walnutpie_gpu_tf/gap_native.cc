// Independent pinned C++ reference for initialization and controller statistics.
#define main original_oracle_main
#include "oracle.cc"
#undef main
#include <walnutpie/config.hpp>
#include <walnutpie/online_moments.hpp>
int main(){
 std::cout<<std::setprecision(17);
 std::vector<Vec> q(3,Vec(2));q[0]<<.2,.4;q[1]<<-.5,.3;q[2]<<.7,-.8;
 Target target{0,2};
 auto c=walnutpie::InitConfigBuilder(3,2).positions(q).masses(target,1e-5,true).build();
 std::cout<<"{\"mass\":";vec(c.mass(0));std::cout<<",\"steps\":[";
 for(int i=0;i<3;++i){Tape tape{17+i};double e=walnutpie::detail::adapt_step(tape,target,q[i],c.mass(i),.1,2);if(i)std::cout<<',';std::cout<<e;}
 Vec eps(3);eps<<.2,.25,.18;Eigen::MatrixXd masses(3,2);masses<<1,2,1.5,2.3,.9,2.2;
 Vec geo=masses.array().log().colwise().mean().exp().matrix();double ge=std::exp(eps.array().log().mean()),md=0,sd=0;
 for(int i=0;i<3;++i){Vec m=masses.row(i).transpose();md=std::max(md,walnutpie::detail::l2_rel_diff(m,geo));sd=std::max(sd,std::abs(eps[i]-ge)/ge);}
 Vec means(3),variances(3);means<<-2,-2.1,-1.9;variances<<1,1.2,.8;
 std::cout<<"],\"warmup_statistics\":["<<sd<<','<<md<<"],\"sampling_statistic\":"<<std::sqrt(1+walnutpie::detail::variance(means)/variances.mean())<<"}\n";
}
