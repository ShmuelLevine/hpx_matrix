#include <cstdlib>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <iostream>

#include <chrono>
#include <hpx/lcos/wait_all.hpp>
#include <random>

int main(int argc, char **argv) { return hpx::init(argc, argv); }

int hpx_main(int argc, char **argv) {

  static constexpr int NP = 1500;
  static constexpr int Nvars = 150000;
  static constexpr int var_range = 1000;
  std::vector<std::mt19937> RNGs;

  std::random_device rd;
  RNGs.reserve(NP);
  for (int i = 0; i < NP; ++i)
    RNGs.emplace_back(rd());

  std::vector<hpx::future<std::vector<int>>> CR_vectors;
  std::vector<std::vector<int>> CR_vectors_2;

  CR_vectors.reserve(NP);
  CR_vectors_2.resize(NP);
  for (auto &vec : CR_vectors_2)
    vec.resize(Nvars); // Since this vector will be used repeatedly for
                       // different execution policies, and default
                       // constructor for int is trivial (much moreso than
                       // hpx::future<std::vector<int>>) for consistency,
                       // use std::vector::resize() here rather than
                       // std::vector<int>::reserve();

  std::cout << "Case 1:       Outer: execution::par\n\
              Inner: execution::par\n\n";

  auto start1 = std::chrono::high_resolution_clock::now();
  hpx::parallel::for_loop_n(
      hpx::parallel::execution::par, 0, NP,
      [=, &RNGs, &CR_vectors_2](int const &pop_idx) {
        hpx::parallel::for_loop_n(
            hpx::parallel::execution::par, 0, Nvars,
            [=, &RNGs, &CR_vectors_2](int const &vec_idx) {
              std::uniform_int_distribution<int> dist(-var_range, var_range);
              CR_vectors_2[pop_idx][vec_idx] = dist(RNGs[pop_idx]);
            });
      });

  auto stop1 = std::chrono::high_resolution_clock::now();


  std::cout << "Generate " << NP << " vectors of " << Nvars << " ints\n\n\n";

  std::cout << "Results:\n\n";

  std::cout << "Case 1: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop1 -
                                                                     start1)
                   .count()
            << " μs\n";

  return hpx::finalize();
}
