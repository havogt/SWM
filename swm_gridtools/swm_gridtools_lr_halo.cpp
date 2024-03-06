#include "config.hpp"

#include "gridtools/common/halo_descriptor.hpp"
#include "initial_conditions.hpp"
#include "verify.hpp"
#include <gridtools/common/defs.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/sid/sid_shift_origin.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <iostream>

namespace gt = gridtools;
namespace st = gt::stencil;

#define TRUE 1
#define FALSE 0
#define MM 16
#define NN 16
#define M_LEN (M + 1)
#define N_LEN (N + 1)
#define SIZE ((M_LEN) * (N_LEN))
#define ITMAX 4000
#define L_OUT TRUE
#define VAL_OUT TRUE
#define VAL_DEEP TRUE
#define VAL_WITH_HALO                                                          \
  FALSE // if true we check also the halo which is included in the verification
        // files (from shallow_swap.c)

void boundary(auto &u) {
  int M = u->lengths()[0] - 2;
  int N = u->lengths()[1] - 2;
  auto u_v = u->host_view();
  for (std::size_t j = 1; j < N + 1; ++j) {
    u_v(0, j) = u_v(M, j);
    u_v(M + 1, j) = u_v(1, j);
  }
  for (std::size_t i = 1; i < M + 1; ++i) {
    u_v(i, N + 1) = u_v(i, 1);
    u_v(i, 0) = u_v(i, N);
  }
  u_v(0, 0) = u_v(M, N);
  u_v(M + 1, 0) = u_v(1, N);
  u_v(M + 1, N + 1) = u_v(1, 1);
  u_v(0, N + 1) = u_v(M, 1);
}

struct calc_cucvzh {
  using fsdx = st::cartesian::in_accessor<0>;
  using fsdy = st::cartesian::in_accessor<1>;
  using u = st::cartesian::in_accessor<2, st::extent<0, 1, -1, 0>>;
  using v = st::cartesian::in_accessor<3, st::extent<-1, 0, 0, 1>>;
  using p = st::cartesian::in_accessor<4, st::extent<-1, 1, -1, 1>>;
  using cu = st::cartesian::inout_accessor<5>;
  using cv = st::cartesian::inout_accessor<6>;
  using z = st::cartesian::inout_accessor<7>;
  using h = st::cartesian::inout_accessor<8>;

  using param_list = st::make_param_list<fsdx, fsdy, u, v, p, cu, cv, z, h>;

  template <class Eval> GT_FUNCTION static void apply(Eval &&eval) {
    eval(cu{}) = 0.5 * (eval(p{}) + eval(p{-1, 0, 0})) * eval(u{});
    eval(cv{}) = 0.5 * (eval(p{}) + eval(p{0, -1, 0})) * eval(v{});
    eval(z{}) = ((eval(fsdx{}) * (eval(v{}) - eval(v{-1, 0, 0}))) -
                 eval(fsdy{}) * (eval(u{}) - eval(u{0, -1, 0}))) /
                (eval(p{-1, -1, 0}) + eval(p{0, -1, 0}) + eval(p{}) +
                 eval(p{-1, 0, 0}));
    eval(h{}) =
        eval(p{}) +
        0.25 * (eval(u{1, 0, 0}) * eval(u{1, 0, 0}) + eval(u{}) * eval(u{}) +
                eval(v{0, 1, 0}) * eval(v{0, 1, 0}) + eval(v{}) * eval(v{}));
  }
};

struct calc_uvp {
  using tdts8 = st::cartesian::in_accessor<0>;
  using tdtsdx = st::cartesian::in_accessor<1>;
  using tdtsdy = st::cartesian::in_accessor<2>;
  using uold = st::cartesian::in_accessor<3>;
  using vold = st::cartesian::in_accessor<4>;
  using pold = st::cartesian::in_accessor<5>;
  using cu = st::cartesian::in_accessor<6, st::extent<-1, 1, -1, 1>>;
  using cv = st::cartesian::in_accessor<7, st::extent<-1, 1, -1, 1>>;
  using z = st::cartesian::in_accessor<8, st::extent<-1, 1, -1, 1>>;
  using h = st::cartesian::in_accessor<9, st::extent<-1, 1, -1, 1>>;
  using unew = st::cartesian::inout_accessor<10>;
  using vnew = st::cartesian::inout_accessor<11>;
  using pnew = st::cartesian::inout_accessor<12>;

  using param_list = st::make_param_list<tdts8, tdtsdx, tdtsdy, uold, vold,
                                         pold, cu, cv, z, h, unew, vnew, pnew>;

  template <class Eval> GT_FUNCTION static void apply(Eval &&eval) {
    eval(unew{}) = eval(uold{}) +
                   eval(tdts8{}) * (eval(z{0, 1, 0}) + eval(z{})) *
                       (eval(cv{0, 1, 0}) + eval(cv{}) + eval(cv{-1, 1, 0}) +
                        eval(cv{-1, 0, 0})) -
                   eval(tdtsdx{}) * (eval(h{}) - eval(h{-1, 0, 0}));
    eval(vnew{}) = eval(vold{}) -
                   eval(tdts8{}) * (eval(z{1, 0, 0}) + eval(z{})) *
                       (eval(cu{1, 0, 0}) + eval(cu{1, -1, 0}) + eval(cu{}) +
                        eval(cu{0, -1, 0})) -
                   eval(tdtsdy{}) * (eval(h{}) - eval(h{0, -1, 0}));
    eval(pnew{}) = eval(pold{}) -
                   eval(tdtsdx{}) * (eval(cu{1, 0, 0}) - eval(cu{})) -
                   eval(tdtsdy{}) * (eval(cv{0, 1, 0}) - eval(cv{}));
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
  std::array<std::array<std::size_t, 2>, 2> u_halo_verify = {{{1, 1}, {1, 1}}};
  std::array<std::array<std::size_t, 2>, 2> v_halo_verify = {{{1, 1}, {1, 1}}};
  std::array<std::array<std::size_t, 2>, 2> p_halo_verify = {{{1, 1}, {1, 1}}};
  std::array<std::array<std::size_t, 2>, 2> z_halo_verify = {{{1, 1}, {1, 1}}};

  double dx = 100000.;
  double dy = 100000.;
  double a = 1000000.;
  double dt = 90.;
  double alpha = 0.001;

  double fsdx = 4. / dx;
  double fsdy = 4. / dy;

  auto storage_builder =
      gt::storage::builder<storage_traits_t>.dimensions(MM + 2, NN + 2).type<double>();

  auto [u, v, p] = initial_conditions(storage_builder, MM, NN, dx, dy, a);

  auto grid = st::make_grid(gt::halo_descriptor(1, 1, 1, MM, MM + 2),
                            gt::halo_descriptor(1, 1, 1, NN, NN + 2), 1);
  auto full_grid = st::make_grid(MM + 2, NN + 2, 1);

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
      if (verify_uvp(u, v, p, MM, NN, step, "init", u_halo_verify,
                     v_halo_verify, p_halo_verify))
        std::cout << "step " << step << " init passed" << std::endl;
      else
        exit(1);
    }

    st::run(calc_cucvzh_spec, stencil_backend_t(), grid,
            st::global_parameter(fsdx), st::global_parameter(fsdy), u, v, p, cu,
            cv, z, h);

    boundary(cu);
    boundary(cv);
    boundary(h);
    boundary(z);

    if (VAL_DEEP && step <= 1) {
      if (verify_cucvzh(cu, cv, z, h, MM, NN, step, "t100", u_halo_verify,
                        v_halo_verify, z_halo_verify, p_halo_verify))
        std::cout << "step " << step << " t100 passed" << std::endl;
      else
        exit(1);
    }

    auto tdts8 = tdt / 8.;
    auto tdtsdx = tdt / dx;
    auto tdtsdy = tdt / dy;

    st::run(calc_uvp_spec, stencil_backend_t(), grid,
            st::global_parameter(tdts8), st::global_parameter(tdtsdx),
            st::global_parameter(tdtsdy), uold, vold, pold, cu, cv, z, h, unew,
            vnew, pnew);

    boundary(unew);
    boundary(vnew);
    boundary(pnew);

    if (VAL_DEEP && step <= 1) {
      if (verify_uvp(unew, vnew, pnew, MM, NN, step, "t200", u_halo_verify,
                     v_halo_verify, p_halo_verify))
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
