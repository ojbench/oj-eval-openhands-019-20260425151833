#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    /*
     *
     *
     *
     *
     *
     *
     * YOUR CODE HERE
     *
     *
     *
     *
     *
     *
     */
    // Move query, keys[0..i], values[0..i] to Shared Memory
    gpu_sim.MoveMatrixToSharedMem(current_query);
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);
    }

    // Build K_stack by concatenating keys along rows (axis=0)
    Matrix *k_stack = matrix_memory_allocator.Allocate("k_stack");
    if (i == 0) {
      gpu_sim.Copy(keys[0], k_stack, kInSharedMemory);
    } else {
      Matrix *cur = matrix_memory_allocator.Allocate("k_cur");
      gpu_sim.Concat(keys[0], keys[1], cur, 0, kInSharedMemory);
      for (size_t j = 2; j <= i; ++j) {
        Matrix *next_mat = matrix_memory_allocator.Allocate("k_step");
        gpu_sim.Concat(cur, keys[j], next_mat, 0, kInSharedMemory);
        cur = next_mat;
      }
      gpu_sim.Copy(cur, k_stack, kInSharedMemory);
    }

    // Build V_stack similarly
    Matrix *v_stack = matrix_memory_allocator.Allocate("v_stack");
    if (i == 0) {
      gpu_sim.Copy(values[0], v_stack, kInSharedMemory);
    } else {
      Matrix *curv = matrix_memory_allocator.Allocate("v_cur");
      gpu_sim.Concat(values[0], values[1], curv, 0, kInSharedMemory);
      for (size_t j = 2; j <= i; ++j) {
        Matrix *next_v = matrix_memory_allocator.Allocate("v_step");
        gpu_sim.Concat(curv, values[j], next_v, 0, kInSharedMemory);
        curv = next_v;
      }
      gpu_sim.Copy(curv, v_stack, kInSharedMemory);
    }

    // K^T
    gpu_sim.Transpose(k_stack, kInSharedMemory);

    // Scores = Q * K^T
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_stack, scores);

    // Softmax over rows: exp then divide by row sums
    Matrix *scores_exp = matrix_memory_allocator.Allocate("scores_exp");
    gpu_sim.MatExp(scores, scores_exp);

    Matrix *softmax_mat = matrix_memory_allocator.Allocate("softmax");
    bool softmax_initialized = false;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row_r = matrix_memory_allocator.Allocate("row_r");
      gpu_sim.GetRow(scores_exp, r, row_r, kInSharedMemory);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_r, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_r, row_sum, row_soft);
      if (!softmax_initialized) {
        gpu_sim.Copy(row_soft, softmax_mat, kInSharedMemory);
        softmax_initialized = true;
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_step");
        gpu_sim.Concat(softmax_mat, row_soft, new_softmax, 0, kInSharedMemory);
        softmax_mat = new_softmax;
      }
    }

    // Answer = softmax * V_stack
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax_mat, v_stack, answer);

    // Move answer to HBM for committing later
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Execute the queued instructions and then commit the answer
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu