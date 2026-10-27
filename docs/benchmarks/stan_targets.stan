data {
  int<lower=1> d;
  int<lower=0,upper=4> kind;
  matrix[d,d] inverse_transform;
  int<lower=0> n;
  matrix[n,d] design;
  real bend;
}
parameters { vector[d] q; }
model {
  if (kind == 0) {
    target += -0.5 * dot_self(q);
  } else if (kind == 1) {
    target += -0.5 * dot_self(inverse_transform * q);
  } else if (kind == 2) {
    for (j in 1:(d %/% 2)) {
      real x=q[2*j-1];
      real y=q[2*j]-bend*(square(x)-1);
      target += -0.5*(square(x)+square(y));
    }
  } else if (kind == 3) {
    target += -0.5*square(q[1]/3) -0.5*(d-1)*q[1];
    target += -0.5*dot_self(tail(q,d-1))*exp(-q[1]);
  } else {
    target += -0.5*dot_self(q)-0.5*dot_self(design*q)/n;
  }
}
