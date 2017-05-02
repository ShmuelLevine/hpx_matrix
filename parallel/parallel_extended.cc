#include <cstdlib>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/parallel/algorithms/transform_reduce_binary.hpp>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include <chrono>
#include <gtest/gtest.h>
#include <hpx/lcos/wait_all.hpp>
#include <random>

int main(int argc, char **argv) { return hpx::init(argc, argv); }

int hpx_main(int argc, char **argv) {

  using CR_vec_type = std::vector<int>;
  using CR_vec_future_set = std::vector<hpx::future<CR_vec_type>>;
  using CR_vec_set = std::vector<CR_vec_type>;

  static constexpr int NP = 1500;
  static constexpr int Nvars = 150000;
  static constexpr int var_range = 1000;
  std::vector<std::mt19937> RNGs;

  std::random_device rd;
  RNGs.reserve(NP);
  for (int i = 0; i < NP; ++i)
    RNGs.emplace_back(rd());
  //  std::uniform_int_distribution<int> dist(-var_range, var_range);

  CR_vec_future_set CR_vectors;
  CR_vectors.reserve(NP);

  CR_vec_set CR_vectors_2;
  CR_vectors_2.resize(NP);
  for (auto &vec : CR_vectors_2)
    vec.resize(Nvars); // Since this vector will be used repeatedly for
                       // different execution policies, and default
                       // constructor for int is trivial (much moreso than
                       // hpx::future<std::vector<int>>) for consistency,
                       // use std::vector::resize() here rather than
                       // std::vector<int>::reserve();

  /* ============================================
     Case 1: hpx::async()
     ============================================
  */
  auto start1 = std::chrono::high_resolution_clock::now();
  std::cout << "Generate random int vectors using hpx::async and wait_all\n";
  for (int i = 0; i < NP; ++i) {
    CR_vectors.push_back(hpx::async(hpx::launch::async, [=, &RNGs]() mutable {
      CR_vec_type cr_vec;
      cr_vec.reserve(Nvars);
      std::uniform_int_distribution<int> dist(-var_range, var_range);

      for (int j = 0; j < Nvars; ++j)
        cr_vec.push_back(dist(RNGs[i]));
      return cr_vec;

    }));
  }
  hpx::wait_all(CR_vectors);
  std::cout << "Finished generating random int vectors with hpx::async\n";
  auto stop1 = std::chrono::high_resolution_clock::now();

  /* ============================================
     Case 2:  Outer: execution::seq
              Inner: execution::seq
     ============================================
  */
  std::cout << "Case 2:       Outer: execution::seq\n\
              Inner: execution::seq\n\n";

  hpx::parallel::for_loop_n(
      hpx::parallel::execution::seq, 0, NP,
      [=, &RNGs, &CR_vectors_2](int const &pop_idx) {
        hpx::parallel::for_loop_n(
            hpx::parallel::execution::seq, 0, Nvars,
            [=, &RNGs, &CR_vectors_2](int const &vec_idx) {
              std::uniform_int_distribution<int> dist(-var_range, var_range);
              CR_vectors_2[pop_idx][vec_idx] = dist(RNGs[pop_idx]);
            });
      });

  auto stop2 = std::chrono::high_resolution_clock::now();

  /* ============================================
     Case 3:  Outer: execution::seq
              Inner: execution::par
     ============================================
  */
  std::cout << "Case 3:       Outer: execution::seq\n\
              Inner: execution::par\n\n";

  hpx::parallel::for_loop_n(
      hpx::parallel::execution::seq, 0, NP,
      [=, &RNGs, &CR_vectors_2](int const &pop_idx) {
        hpx::parallel::for_loop_n(
            hpx::parallel::execution::par, 0, Nvars,
            [=, &RNGs, &CR_vectors_2](int const &vec_idx) {
              std::uniform_int_distribution<int> dist(-var_range, var_range);
              CR_vectors_2[pop_idx][vec_idx] = dist(RNGs[pop_idx]);
            });
      });

  auto stop3 = std::chrono::high_resolution_clock::now();

  /* ============================================
     Case 4:  Outer: execution::par
              Inner: execution::par
     ============================================
  */
  std::cout << "Case 4:       Outer: execution::par\n\
              Inner: execution::par\n\n";

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

  auto stop4 = std::chrono::high_resolution_clock::now();

  /* ============================================
     Case 5:  Outer: execution::seq
              Inner: execution::par_unseq
     ============================================
  */
  std::cout << "Case 5:       Outer: execution::seq\n\
              Inner: execution::par_unseq\n\n";

  hpx::parallel::for_loop_n(
      hpx::parallel::execution::seq, 0, NP,
      [=, &RNGs, &CR_vectors_2](int const &pop_idx) {
        hpx::parallel::for_loop_n(
            hpx::parallel::execution::par_unseq, 0, Nvars,
            [=, &RNGs, &CR_vectors_2](int const &vec_idx) {
              std::uniform_int_distribution<int> dist(-var_range, var_range);
              CR_vectors_2[pop_idx][vec_idx] = dist(RNGs[pop_idx]);
            });
      });

  auto stop5 = std::chrono::high_resolution_clock::now();

  /* ============================================
     Case 6:  Outer: execution::par
              Inner: execution::par_unseq
     ============================================
  */
  std::cout << "Case 6:       Outer: execution::par\n\
              Inner: execution::par_unseq\n\n";

  hpx::parallel::for_loop_n(
      hpx::parallel::execution::par, 0, NP,
      [=, &RNGs, &CR_vectors_2](int const &pop_idx) {
        hpx::parallel::for_loop_n(
            hpx::parallel::execution::par_unseq, 0, Nvars,
            [=, &RNGs, &CR_vectors_2](int const &vec_idx) {
              std::uniform_int_distribution<int> dist(-var_range, var_range);
              CR_vectors_2[pop_idx][vec_idx] = dist(RNGs[pop_idx]);
            });
      });

  auto stop6 = std::chrono::high_resolution_clock::now();

  /* ============================================
     Case 7:  Outer: execution::par_unseq
     Inner: execution::par_unseq
     ============================================
  */
  std::cout << "Case 7:       Outer: execution::par_unseq\n\
              Inner: execution::par_unseq\n\n";

  hpx::parallel::for_loop_n(
      hpx::parallel::execution::par_unseq, 0, NP,
      [=, &RNGs, &CR_vectors_2](int const &pop_idx) {
        hpx::parallel::for_loop_n(
            hpx::parallel::execution::par_unseq, 0, Nvars,
            [=, &RNGs, &CR_vectors_2](int const &vec_idx) {
              std::uniform_int_distribution<int> dist(-var_range, var_range);
              CR_vectors_2[pop_idx][vec_idx] = dist(RNGs[pop_idx]);
            });
      });

  auto stop7 = std::chrono::high_resolution_clock::now();

  std::cout << "Generate " << NP << " vectors of " << Nvars << " ints\n\n\n";

  std::cout << "Results:\n\n"
            << "Case 1: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop1 -
                                                                     start1)
                   .count()
            << " μs \n";

  std::cout << "Case 2: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop2 -
                                                                     stop1)
                   .count()
            << " μs \n";

  std::cout << "Case 3: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop3 -
                                                                     stop2)
                   .count()
            << " μs \n";

  std::cout << "Case 4: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop4 -
                                                                     stop3)
                   .count()
            << " μs \n";

  std::cout << "Case 5: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop5 -
                                                                     stop4)
                   .count()
            << " μs \n";

  std::cout << "Case 6: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop6 -
                                                                     stop5)
                   .count()
            << " μs \n";

  std::cout << "Case 7: "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop7 -
                                                                     stop6)
                   .count()
            << " μs \n";

  return hpx::finalize();
}
