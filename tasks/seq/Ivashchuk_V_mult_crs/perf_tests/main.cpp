#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(Ivashchuk_V_sparse_matrix_seq, TestPipelineRun) {
  constexpr int kCount = 3000;  // Increased to 3000

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> value_dis(-2.0, 2.0);

  // Create dense matrices to ensure long computation time
  for (int i = 0; i < kCount; ++i) {
    // Main diagonal with complex values
    in1[i * kCount + i] = std::complex<double>(2.0, 1.0);
    in2[i * kCount + i] = std::complex<double>(1.5, -0.5);

    // Multiple adjacent diagonals
    for (int offset = 1; offset <= 10; ++offset) {  // Increased to 10
      if (i >= offset) {
        in1[i * kCount + (i - offset)] = std::complex<double>(0.3, 0.2);
        in2[i * kCount + (i - offset)] = std::complex<double>(0.2, 0.3);
      }
      if (i < kCount - offset) {
        in1[i * kCount + (i + offset)] = std::complex<double>(0.4, -0.1);
        in2[i * kCount + (i + offset)] = std::complex<double>(-0.1, 0.4);
      }
    }

    // Many random elements in each row
    for (int j = 0; j < 30; ++j) {  // Increased to 30
      int random_col = std::uniform_int_distribution<>(0, kCount - 1)(gen);
      in1[i * kCount + random_col] = std::complex<double>(value_dis(gen), value_dis(gen));
      in2[i * kCount + random_col] = std::complex<double>(value_dis(gen), value_dis(gen));
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Check result correctness
  bool has_non_zero = false;
  for (size_t i = 0; i < kCount * kCount; ++i) {
    if (std::abs(out[i].real()) > 1e-10 || std::abs(out[i].imag()) > 1e-10) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestTaskRun) {
  constexpr int kCount = 4000;  // Maximum size - 4000x4000

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> value_dis(-2.0, 2.0);

  // Create extremely dense matrices to ensure long computation time
  for (int i = 0; i < kCount; ++i) {
    // Main diagonal with complex values
    in1[i * kCount + i] = std::complex<double>(2.0, 1.0);
    in2[i * kCount + i] = std::complex<double>(1.5, -0.5);

    // Multiple adjacent diagonals
    for (int offset = 1; offset <= 20; ++offset) {  // Maximum - 20 diagonals
      if (i >= offset) {
        in1[i * kCount + (i - offset)] = std::complex<double>(0.3, 0.2);
        in2[i * kCount + (i - offset)] = std::complex<double>(0.2, 0.3);
      }
      if (i < kCount - offset) {
        in1[i * kCount + (i + offset)] = std::complex<double>(0.4, -0.1);
        in2[i * kCount + (i + offset)] = std::complex<double>(-0.1, 0.4);
      }
    }

    // Maximum random elements in each row
    for (int j = 0; j < 50; ++j) {  // Maximum - 50 random elements per row
      int random_col = std::uniform_int_distribution<>(0, kCount - 1)(gen);
      in1[i * kCount + random_col] = std::complex<double>(value_dis(gen), value_dis(gen));
      in2[i * kCount + random_col] = std::complex<double>(value_dis(gen), value_dis(gen));
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Check result correctness
  bool has_non_zero = false;
  for (size_t i = 0; i < kCount * kCount; ++i) {
    if (std::abs(out[i].real()) > 1e-10 || std::abs(out[i].imag()) > 1e-10) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}