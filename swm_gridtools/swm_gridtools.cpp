#include <cmath>
#include <cstdint>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/sid/sid_shift_origin.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <iostream>

// this include only works if the full gridtools repo is available
#include <gridtools/../../tests/include/verifier.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil/gpu.hpp>
#include <gridtools/storage/gpu.hpp>
using stencil_backend_t = gridtools::stencil::gpu<>;
using storage_traits_t = gridtools::storage::gpu;
#else
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
using storage_traits_t = gridtools::storage::cpu_ifirst;
#endif

namespace gt = gridtools;
namespace st = gt::stencil;

#define TRUE 1
#define FALSE 0
#define MM 16
#define NN 16
#define M_LEN (M + 1)
#define N_LEN (N + 1)
#define SIZE ((M_LEN) * (N_LEN))
#define ITMAX 3
#define L_OUT TRUE
#define VAL_OUT TRUE
#define VAL_DEEP TRUE

void u_boundary(auto &u) {
  int M = u->lengths()[0] - 1;
  int N = u->lengths()[1] - 1;
  auto u_v = u->host_view();
  for (std::size_t j = 0; j < N; ++j) {
    u_v(0, j) = u_v(M, j);
  }
  for (std::size_t i = 0; i < M; ++i) {
    u_v(i + 1, N) = u_v(i + 1, 0);
  }
  u_v(0, N) = u_v(M, 0);
}

void v_boundary(auto &v) {
  int M = v->lengths()[0] - 1;
  int N = v->lengths()[1] - 1;
  auto v_v = v->host_view();
  for (std::size_t j = 0; j < N; ++j) {
    v_v(M, j + 1) = v_v(0, j + 1);
  }

  for (std::size_t i = 0; i < M; ++i) {
    v_v(i, 0) = v_v(i, N);
  }

  v_v(M, 0) = v_v(0, N);
}

void p_boundary(auto &p) {
  int M = p->lengths()[0] - 1;
  int N = p->lengths()[1] - 1;
  auto p_v = p->host_view();
  for (std::size_t j = 0; j < N; ++j) {
    p_v(M, j) = p_v(0, j);
  }

  for (std::size_t i = 0; i < M; ++i) {
    p_v(i, N) = p_v(i, 0);
  }

  p_v(M, N) = p_v(0, 0);
}

void z_boundary(auto &z) {
  int M = z->lengths()[0] - 1;
  int N = z->lengths()[1] - 1;
  auto z_v = z->host_view();
  for (std::size_t j = 0; j < N; ++j) {
    z_v(0, j + 1) = z_v(M, j + 1);
  }
  for (std::size_t i = 0; i < M; ++i) {
    z_v(i + 1, 0) = z_v(i + 1, N);
  }
  z_v(0, 0) = z_v(M, N);
}

auto initialize(auto const &storage_builder, std::size_t M, std::size_t N,
                double dx, double dy, double a) {
  double pi = 4.0 * std::atan(1.0);
  double tpi = 2.0 * pi;
  double d_i = tpi / M;
  double d_j = tpi / N;
  double el = N * dx;
  double pcf = (pi * pi * a * a) / (el * el);

  auto psi = storage_builder.build();
  auto p = storage_builder.build();
  auto psi_v = psi->host_view();
  auto p_v = p->host_view();
  for (std::size_t i = 0; i < M + 1; ++i) {
    for (std::size_t j = 0; j < N + 1; ++j) {
      psi_v(i, j) = a * std::sin((i + 0.5) * d_i) * std::sin((j + 0.5) * d_j);
      p_v(i, j) =
          pcf * (std::cos(2. * i * d_i) + std::cos(2. * j * d_j)) + 50000.;
    }
  }

  auto u = storage_builder.build();
  auto v = storage_builder.build();
  auto u_v = u->host_view();
  auto v_v = v->host_view();
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      u_v(i + 1, j) = -(psi_v(i + 1, j + 1) - psi_v(i + 1, j)) / dy;
      v_v(i, j + 1) = (psi_v(i + 1, j + 1) - psi_v(i, j + 1)) / dx;
    }

  u_boundary(u);
  v_boundary(v);
  return std::tuple(u, v, p);
}

auto read_from_file(char const *filename, std::size_t M, std::size_t N) {
  auto ds =
      gt::storage::builder<storage_traits_t>.dimensions(M + 1, N + 1).type<double>().build();
  auto view = ds->host_view();

  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    printf("Error opening file %s\n", filename);
    exit(1);
  }
  double tmp;
  for (int i = 0; i < M + 1; i++) {
    for (int j = 0; j < N + 1; j++) {
      fread(&tmp, sizeof(double), 1, file);
      view(i, j) = tmp;
    }
  }
  fclose(file);
  return ds;
}

auto read_uvp(int step, std::string const &suffix, std::size_t M,
              std::size_t N) {
  auto storage_builder =
      gt::storage::builder<storage_traits_t>.dimensions(M + 1, N + 1).type<double>();
  auto u = read_from_file(
      ("ref/u.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);
  auto v = read_from_file(
      ("ref/v.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);
  auto p = read_from_file(
      ("ref/p.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);
  return std::tuple(u, v, p);
}

auto verify_uvp(auto const &u, auto const &v, auto const &p, std::size_t M,
                std::size_t N, int step, std::string const &suffix) {
  auto [u_ref, v_ref, p_ref] = read_uvp(step, suffix, M, N);

  if (!gt::verify_data_store(u_ref, u,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in u" << std::endl;
    return false;
  }
  if (!gt::verify_data_store(v_ref, v,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in v" << std::endl;
    return false;
  }
  if (!gt::verify_data_store(p_ref, p,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in p" << std::endl;
    return false;
  }
  return true;
}

auto verify_cucvzh(auto const &cu, auto const &cv, auto const &z, auto const &h,
                   std::size_t M, std::size_t N, int step,
                   std::string const &suffix) {
  auto cu_ref = read_from_file(
      ("ref/cu.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);
  auto cv_ref = read_from_file(
      ("ref/cv.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);
  auto z_ref = read_from_file(
      ("ref/z.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);
  auto h_ref = read_from_file(
      ("ref/h.step" + std::to_string(step) + "." + suffix + ".bin").c_str(), M,
      N);

  if (!gt::verify_data_store(cu_ref, cu,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in cu" << std::endl;
    return false;
  }
  if (!gt::verify_data_store(cv_ref, cv,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in cv" << std::endl;
    return false;
  }
  if (!gt::verify_data_store(z_ref, z,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in z" << std::endl;
    return false;
  }
  if (!gt::verify_data_store(h_ref, h,
                             std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in h" << std::endl;
    return false;
  }
  return true;
}

struct calc_cucvzh {
  using fsdx = st::cartesian::in_accessor<0>;
  using fsdy = st::cartesian::in_accessor<1>;
  using u = st::cartesian::in_accessor<2, st::extent<-1, 0, 0, 1>>;
  using v = st::cartesian::in_accessor<3, st::extent<0, 1, -1, 0>>;
  using p = st::cartesian::in_accessor<4, st::extent<0, 1, 0, 1>>;
  using cu = st::cartesian::inout_accessor<5>;
  using cv = st::cartesian::inout_accessor<6>;
  using z = st::cartesian::inout_accessor<7>;
  using h = st::cartesian::inout_accessor<8>;

  using param_list = st::make_param_list<fsdx, fsdy, u, v, p, cu, cv, z, h>;

  template <class Eval> GT_FUNCTION static void apply(Eval &&eval) {
    eval(cu{}) = 0.5 * (eval(p{1, 0, 0}) + eval(p{})) * eval(u{});
    eval(cv{}) = 0.5 * (eval(p{0, 1, 0}) + eval(p{})) * eval(v{});
    eval(z{}) =
        ((eval(fsdx{}) * (eval(v{1, 0, 0}) - eval(v{}))) -
         eval(fsdy{}) * (eval(u{0, 1, 0}) - eval(u{}))) /
        (eval(p{1, 1, 0}) + eval(p{0, 1, 0}) + eval(p{}) + eval(p{1, 0, 0}));
    eval(h{}) =
        eval(p{}) +
        0.25 * (eval(u{-1, 0, 0}) * eval(u{-1, 0, 0}) + eval(u{}) * eval(u{}) +
                eval(v{0, -1, 0}) * eval(v{0, -1, 0}) + eval(v{}) * eval(v{}));
  }
};

struct calc_uvp {
  using tdts8 = st::cartesian::in_accessor<0>;
  using tdtsdx = st::cartesian::in_accessor<1>;
  using tdtsdy = st::cartesian::in_accessor<2>;
  using uold = st::cartesian::in_accessor<3>;
  using vold = st::cartesian::in_accessor<4>;
  using pold = st::cartesian::in_accessor<5>;
  using cu = st::cartesian::in_accessor<6, st::extent<-1, 0, 0, 1>>;
  using cv = st::cartesian::in_accessor<7, st::extent<0, 1, -1, 0>>;
  using z = st::cartesian::in_accessor<8, st::extent<-1, 0, -1, 0>>;
  using h = st::cartesian::in_accessor<9, st::extent<0, 1, 0, 1>>;
  using unew = st::cartesian::inout_accessor<10>;
  using vnew = st::cartesian::inout_accessor<11>;
  using pnew = st::cartesian::inout_accessor<12>;

  using param_list = st::make_param_list<tdts8, tdtsdx, tdtsdy, uold, vold,
                                         pold, cu, cv, z, h, unew, vnew, pnew>;

  template <class Eval> GT_FUNCTION static void apply(Eval &&eval) {
    eval(unew{}) = eval(uold{}) +
                   eval(tdts8{}) * (eval(z{}) + eval(z{0, -1, 0})) *
                       (eval(cv{1, 0, 0}) + eval(cv{}) + eval(cv{0, -1, 0}) +
                        eval(cv{1, -1, 0})) -
                   eval(tdtsdx{}) * (eval(h{1, 0, 0}) - eval(h{}));
    eval(vnew{}) = eval(vold{}) -
                   eval(tdts8{}) * (eval(z{}) + eval(z{-1, 0, 0})) *
                       (eval(cu{-1, 0, 0}) + eval(cu{-1, 1, 0}) + eval(cu{}) +
                        eval(cu{0, 1, 0})) -
                   eval(tdtsdy{}) * (eval(h{0, 1, 0}) - eval(h{}));
    eval(pnew{}) = eval(pold{}) -
                   eval(tdtsdx{}) * (eval(cu{}) - eval(cu{-1, 0, 0})) -
                   eval(tdtsdy{}) * (eval(cv{}) - eval(cv{0, -1, 0}));
  }
};

struct calc_uvp_old {
  using alpha = st::cartesian::in_accessor<0>;
  using u = st::cartesian::in_accessor<1>;
  using unew = st::cartesian::in_accessor<2>;
  using uold = st::cartesian::inout_accessor<3>;
  using v = st::cartesian::in_accessor<4>;
  using vnew = st::cartesian::in_accessor<5>;
  using vold = st::cartesian::inout_accessor<6>;
  using p = st::cartesian::in_accessor<7>;
  using pnew = st::cartesian::in_accessor<8>;
  using pold = st::cartesian::inout_accessor<9>;

  using param_list =
      st::make_param_list<alpha, u, unew, uold, v, vnew, vold, p, pnew, pold>;

  template <class Eval> GT_FUNCTION static void apply(Eval &&eval) {
    eval(uold{}) = eval(u{}) + eval(alpha{}) * (eval(unew{}) - 2. * eval(u{}) +
                                                eval(uold{}));
    eval(vold{}) = eval(v{}) + eval(alpha{}) * (eval(vnew{}) - 2. * eval(v{}) +
                                                eval(vold{}));
    eval(pold{}) = eval(p{}) + eval(alpha{}) * (eval(pnew{}) - 2. * eval(p{}) +
                                                eval(pold{}));
  }
};

struct copy {
  using in = st::cartesian::in_accessor<0>;
  using out = st::cartesian::inout_accessor<1>;

  using param_list = st::make_param_list<in, out>;

  template <class Eval> GT_FUNCTION static void apply(Eval &&eval) {
    eval(out{}) = eval(in{});
  }
};

int main() {
  double dx = 100000.;
  double dy = 100000.;
  double a = 1000000.;
  double dt = 90.;
  double alpha = 0.001;

  double fsdx = 4. / dx;
  double fsdy = 4. / dy;

  auto storage_builder =
      gt::storage::builder<storage_traits_t>.dimensions(MM + 1, NN + 1).type<double>();

  auto [u, v, p] = initialize(storage_builder, MM, NN, dx, dy, a);

  auto grid = st::make_grid(MM, NN, 1);
  auto full_grid = st::make_grid(MM + 1, NN + 1, 1);

  auto copy_spec = [](auto in, auto out) {
    return st::execute_parallel().stage(copy(), in, out);
  };

  auto calc_cucvzh_spec = [](auto fsdx, auto fsdy, auto u, auto v, auto p,
                             auto cu, auto cv, auto z, auto h) {
    return st::execute_parallel().stage(calc_cucvzh(), fsdx, fsdy, u, v, p, cu,
                                        cv, z, h);
  };

  auto calc_uvp_spec = [](auto tdts8, auto tdtsdx, auto tdtsdy, auto uold,
                          auto vold, auto pold, auto cu, auto cv, auto z,
                          auto h, auto unew, auto vnew, auto pnew) {
    return st::execute_parallel().stage(calc_uvp(), tdts8, tdtsdx, tdtsdy, uold,
                                        vold, pold, cu, cv, z, h, unew, vnew,
                                        pnew);
  };

  auto calc_uvp_old_spec = [](auto alpha, auto u, auto unew, auto uold, auto v,
                              auto vnew, auto vold, auto p, auto pnew,
                              auto pold) {
    return st::execute_parallel().stage(calc_uvp_old(), alpha, u, unew, uold, v,
                                        vnew, vold, p, pnew, pold);
  };

  auto cu = storage_builder.build();
  auto cv = storage_builder.build();
  auto z = storage_builder.build();
  auto h = storage_builder.build();

  auto uold = storage_builder.build();
  auto vold = storage_builder.build();
  auto pold = storage_builder.build();

  st::run(copy_spec, stencil_backend_t(), full_grid, u, uold);
  st::run(copy_spec, stencil_backend_t(), full_grid, v, vold);
  st::run(copy_spec, stencil_backend_t(), full_grid, p, pold);

  auto unew = storage_builder.build();
  auto vnew = storage_builder.build();
  auto pnew = storage_builder.build();

  using namespace gridtools::literals;
  auto u_origin = gt::tuple(1_c, 0_c);
  auto v_origin = gt::tuple(0_c, 1_c);
  auto z_origin = gt::tuple(1_c, 1_c);
  auto p_origin = gt::tuple(0_c, 0_c);

  auto tdt = dt;

  for (int step = 0; step < ITMAX; ++step) {

    if (VAL_DEEP && step <= 2) {
      if (verify_uvp(u, v, p, MM, NN, step, "init"))
        std::cout << "step " << step << " init passed" << std::endl;
      else
        exit(1);
    }

    st::run(calc_cucvzh_spec, stencil_backend_t(), grid,
            st::global_parameter(fsdx), st::global_parameter(fsdy),
            gt::sid::shift_sid_origin(u, u_origin),
            gt::sid::shift_sid_origin(v, v_origin),
            gt::sid::shift_sid_origin(p, p_origin),
            gt::sid::shift_sid_origin(cu, u_origin),
            gt::sid::shift_sid_origin(cv, v_origin),
            gt::sid::shift_sid_origin(z, z_origin),
            gt::sid::shift_sid_origin(h, p_origin));

    u_boundary(cu);
    v_boundary(cv);
    p_boundary(h);
    z_boundary(z);

    if (VAL_DEEP && step <= 1) {
      if (verify_cucvzh(cu, cv, z, h, MM, NN, step, "t100"))
        std::cout << "step " << step << " t100 passed" << std::endl;
      else
        exit(1);
    }

    auto tdts8 = tdt / 8.;
    auto tdtsdx = tdt / dx;
    auto tdtsdy = tdt / dy;

    st::run(calc_uvp_spec, stencil_backend_t(), grid,
            st::global_parameter(tdts8), st::global_parameter(tdtsdx),
            st::global_parameter(tdtsdy),
            gt::sid::shift_sid_origin(uold, u_origin),
            gt::sid::shift_sid_origin(vold, v_origin),
            gt::sid::shift_sid_origin(pold, p_origin),
            gt::sid::shift_sid_origin(cu, u_origin),
            gt::sid::shift_sid_origin(cv, v_origin),
            gt::sid::shift_sid_origin(z, z_origin),
            gt::sid::shift_sid_origin(h, p_origin),
            gt::sid::shift_sid_origin(unew, u_origin),
            gt::sid::shift_sid_origin(vnew, v_origin),
            gt::sid::shift_sid_origin(pnew, p_origin));

    u_boundary(unew);
    v_boundary(vnew);
    p_boundary(pnew);

    if (VAL_DEEP && step <= 1) {
      if (verify_uvp(unew, vnew, pnew, MM, NN, step, "t200"))
        std::cout << "step " << step << " t200 passed" << std::endl;
      else
        exit(1);
    }

    if (step > 0) {
      st::run(calc_uvp_old_spec, stencil_backend_t(), full_grid,
              st::global_parameter(alpha), u, unew, uold, v, vnew, vold, p,
              pnew, pold);

      st::run(copy_spec, stencil_backend_t(), full_grid, unew, u);
      st::run(copy_spec, stencil_backend_t(), full_grid, vnew, v);
      st::run(copy_spec, stencil_backend_t(), full_grid, pnew, p);
    } else {
      tdt += tdt;

      st::run(copy_spec, stencil_backend_t(), full_grid, u, uold);
      st::run(copy_spec, stencil_backend_t(), full_grid, v, vold);
      st::run(copy_spec, stencil_backend_t(), full_grid, p, pold);

      st::run(copy_spec, stencil_backend_t(), full_grid, unew, u);
      st::run(copy_spec, stencil_backend_t(), full_grid, vnew, v);
      st::run(copy_spec, stencil_backend_t(), full_grid, pnew, p);
    }
  }
}
