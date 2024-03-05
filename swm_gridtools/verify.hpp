#pragma once

#include "config.hpp"

#include <gridtools/storage/builder.hpp>
#include <iostream>
#include <string>
// this include only works if the full gridtools repo is available
#include <gridtools/../../tests/include/verifier.hpp>

auto read_from_file(char const *filename, std::size_t M, std::size_t N) {
  auto ds =
      gridtools::storage::builder<storage_traits_t>.dimensions(M + 1, N + 1).type<double>().build();
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
      gridtools::storage::builder<storage_traits_t>.dimensions(M + 1, N + 1).type<double>();
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

  if (!gridtools::verify_data_store(
          u_ref, u, std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in u" << std::endl;
    return false;
  }
  if (!gridtools::verify_data_store(
          v_ref, v, std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in v" << std::endl;
    return false;
  }
  if (!gridtools::verify_data_store(
          p_ref, p, std::array{std::array{0, 0}, std::array{0, 0}})) {
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

  if (!gridtools::verify_data_store(
          cu_ref, cu, std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in cu" << std::endl;
    return false;
  }
  if (!gridtools::verify_data_store(
          cv_ref, cv, std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in cv" << std::endl;
    return false;
  }
  if (!gridtools::verify_data_store(
          z_ref, z, std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in z" << std::endl;
    return false;
  }
  if (!gridtools::verify_data_store(
          h_ref, h, std::array{std::array{0, 0}, std::array{0, 0}})) {
    std::cout << "in h" << std::endl;
    return false;
  }
  return true;
}
