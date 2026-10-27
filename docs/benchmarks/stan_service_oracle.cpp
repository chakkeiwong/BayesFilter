// Diagnostic oracle: actual Stan initialization service on flat scalar parameters.
#include <stan/services/util/initialize.hpp>
#include <stan/io/array_var_context.hpp>
#include <iostream>
#include <iomanip>
#include <limits>

struct uniform_tape {
  using result_type = double;
  size_t used = 0;
  static double min() { return 0.; }
  static double max() { return 1.; }
  double operator()() {
    // First pair is outside the bounded target; subsequent pairs are inside.
    const double first[] = {.875, .5};
    const double later[] = {.375, .75};
    size_t index = used++;
    return index < 2 ? first[index] : later[index % 2];
  }
};

struct flat_model {
  int kind;
  mutable int attempts = 0;
  int num_params_r() const { return 2; }
  void get_param_names(std::vector<std::string>& names, bool, bool) const {
    names = {"a", "b"};
  }
  void get_dims(std::vector<std::vector<size_t>>& dims, bool, bool) const {
    dims = {{}, {}};
  }
  template <class RNG>
  void write_array(RNG&, std::vector<double>& q, std::vector<int>&,
                   std::vector<double>& out, bool, bool, std::ostream*) const {
    ++attempts;
    out = q;
  }
  void transform_inits(const stan::io::var_context& ctx, std::vector<int>&,
                       std::vector<double>& q, std::ostream*) const {
    q = {ctx.vals_r("a")[0], ctx.vals_r("b")[0]};
  }
  template <bool propto, bool jacobian, class T>
  T log_prob(std::vector<T>& q, std::vector<int>&, std::ostream*) const {
    if (kind == 2 || (kind == 1 && std::abs(stan::math::value_of(q[0])) >= 1.))
      return T(-std::numeric_limits<double>::infinity());
    if (kind == 3) return 1e308 * (q[0] + q[1]);
    return -.5 * (q[0] * q[0] + q[1] * q[1]);
  }
};

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  flat_model model{std::stoi(argv[1])};
  int mask = std::stoi(argv[2]);
  double radius = std::stod(argv[3]);
  std::vector<std::string> names;
  std::vector<double> values;
  std::vector<std::vector<size_t>> dims;
  for (int i = 0; i < 2; ++i) if (mask & (1 << i)) {
    names.push_back(i == 0 ? "a" : "b");
    values.push_back(std::stod(argv[4 + i]));
    dims.push_back({});
  }
  stan::io::array_var_context context(names, values, dims);
  uniform_tape rng;
  stan::callbacks::logger logger;
  stan::callbacks::writer writer;
  std::vector<double> q;
  bool ok = true;
  try {
    q = stan::services::util::initialize(model, context, rng, radius, false,
                                        logger, writer);
  } catch (const std::domain_error&) { ok = false; }
  std::cout << std::setprecision(17) << ok << " " << model.attempts << " " << rng.used;
  for (double x : q) std::cout << " " << x;
  std::cout << "\n";
}
