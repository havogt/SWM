#pragma once

#include <cmath>
#include <cstddef>

template <class T> auto psi(T dx, T dy, T a, int M, int N) {
  T pi = static_cast<T>(4) * std::atan(static_cast<T>(1));
  T tpi = static_cast<T>(2) * pi;
  T d_i = tpi / static_cast<T>(M);
  T d_j = tpi / static_cast<T>(N);
  return [=](int i, int j, auto... dummy) {
    return a * std::sin((static_cast<T>(i) + static_cast<T>(0.5)) * d_i) *
           std::sin((static_cast<T>(j) + static_cast<T>(0.5)) * d_j);
  };
}

template <class T> auto p(T dx, T dy, T a, int M, int N) {
  T pi = static_cast<T>(4) * std::atan(static_cast<T>(1));
  T tpi = static_cast<T>(2) * pi;
  T d_i = tpi / static_cast<T>(M);
  T d_j = tpi / static_cast<T>(N);
  T el = static_cast<T>(N) * dx;
  T pcf = (pi * pi * a * a) / (el * el);
  return [=](int i, int j, auto... dummy) {
    return pcf * (std::cos(static_cast<T>(2) * static_cast<T>(i) * d_i) +
                  std::cos(static_cast<T>(2) * static_cast<T>(j) * d_j)) +
           static_cast<T>(50000);
  };
}

template <class T> auto u(T dx, T dy, T a, int M, int N) {
  auto cpsi = psi(dx, dy, a, M, N);
  return [=](int i, int j, auto... dummy) {
    return -(cpsi(i, j + 1) - cpsi(i, j)) / dy;
  };
}

template <class T> auto v(T dx, T dy, T a, int M, int N) {
  auto cpsi = psi(dx, dy, a, M, N);
  return [=](int i, int j, auto... dummy) {
    return (cpsi(i + 1, j) - cpsi(i, j)) / dx;
  };
}

auto initial_conditions(auto const &storage_builder, std::size_t M,
                        std::size_t N, double dx, double dy, double a) {
  return std::tuple(storage_builder.initializer(u(dx, dy, a, M, N))(),
                    storage_builder.initializer(v(dx, dy, a, M, N))(),
                    storage_builder.initializer(p(dx, dy, a, M, N))());
}
