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
    // Ensure current query is in Shared Memory
    if (current_query->GetPosition() == kInGpuHbm) {
      gpu_sim.MoveMatrixToSharedMem(current_query);
    }

    // Maintain cumulative K, K^T and V stacks across rounds to avoid rebuilding
    static Matrix *k_accum = nullptr;   // shape: (i+1, d)
    static Matrix *kt_accum = nullptr;  // shape: (d, i+1)
    static Matrix *v_accum = nullptr;   // shape: (i+1, d)

    // Move only the newly added key/value to Shared Memory if needed
    if (keys[i]->GetPosition() == kInGpuHbm) gpu_sim.MoveMatrixToSharedMem(keys[i]);
    if (values[i]->GetPosition() == kInGpuHbm) gpu_sim.MoveMatrixToSharedMem(values[i]);

    if (i == 0) {
      k_accum = keys[0];
      v_accum = values[0];
      // initialize kt_accum = transpose(keys[0])
      kt_accum = matrix_memory_allocator.Allocate("kt_accum");
      gpu_sim.Copy(keys[0], kt_accum, kInSharedMemory);
      gpu_sim.Transpose(kt_accum, kInSharedMemory);
    } else {
      // update k_accum by appending a new row
      Matrix *prev_k = k_accum;
      Matrix *new_k = matrix_memory_allocator.Allocate("k_accum");
      gpu_sim.Concat(k_accum, keys[i], new_k, 0, kInSharedMemory);
      k_accum = new_k;
      if (i >= 2) gpu_sim.ReleaseMatrix(prev_k);

      // update kt_accum by appending a new column (transpose of keys[i])
      Matrix *col_i = matrix_memory_allocator.Allocate("k_col");
      gpu_sim.Copy(keys[i], col_i, kInSharedMemory);
      gpu_sim.Transpose(col_i, kInSharedMemory);
      Matrix *prev_kt = kt_accum;
      Matrix *new_kt = matrix_memory_allocator.Allocate("kt_accum");
      gpu_sim.Concat(kt_accum, col_i, new_kt, 1, kInSharedMemory);
      kt_accum = new_kt;
      gpu_sim.ReleaseMatrix(col_i);
      if (i >= 2) gpu_sim.ReleaseMatrix(prev_kt);

      // update v_accum by appending a new row
      Matrix *prev_v = v_accum;
      Matrix *new_v = matrix_memory_allocator.Allocate("v_accum");
      gpu_sim.Concat(v_accum, values[i], new_v, 0, kInSharedMemory);
      v_accum = new_v;
      if (i >= 2) gpu_sim.ReleaseMatrix(prev_v);
    }

    // Build answer row by row to reduce SRAM
    Matrix *answer = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row_q = matrix_memory_allocator.Allocate("row_q");
      gpu_sim.GetRow(current_query, r, row_q, kInSharedMemory);
      Matrix *row_scores = matrix_memory_allocator.Allocate("row_scores");
      gpu_sim.MatMul(row_q, kt_accum, row_scores);
      gpu_sim.ReleaseMatrix(row_q);

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_scores, row_exp);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);
      // release intermediates
      gpu_sim.ReleaseMatrix(row_scores);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);

      // Compute row_out = sum_k row_soft[k] * V[k,:] to reduce time from O((i+1)^2 d) to O((i+1) d)
      Matrix *row_out = nullptr;
      for (size_t k = 0; k <= i; ++k) {
        Matrix *v_row = matrix_memory_allocator.Allocate("v_row");
        gpu_sim.GetRow(v_accum, k, v_row, kInSharedMemory);
        Matrix *factor = matrix_memory_allocator.Allocate("factor");
        gpu_sim.GetColumn(row_soft, k, factor, kInSharedMemory);
        Matrix *scaled = matrix_memory_allocator.Allocate("scaled");
        gpu_sim.MatMulNum(v_row, factor, scaled);
        gpu_sim.ReleaseMatrix(v_row);
        gpu_sim.ReleaseMatrix(factor);
        if (!row_out) {
          row_out = scaled;
        } else {
          Matrix *old_out = row_out;
          Matrix *new_out = matrix_memory_allocator.Allocate("row_out_acc");
          gpu_sim.MatAdd(row_out, scaled, new_out);
          row_out = new_out;
          gpu_sim.ReleaseMatrix(old_out);
          gpu_sim.ReleaseMatrix(scaled);
        }
      }
      gpu_sim.ReleaseMatrix(row_soft);
      if (!answer) {
        answer = row_out;
      } else {
        Matrix *old_ans = answer;
        Matrix *new_ans = matrix_memory_allocator.Allocate("answer_step");
        gpu_sim.Concat(answer, row_out, new_ans, 0, kInSharedMemory);
        answer = new_ans;
        gpu_sim.ReleaseMatrix(old_ans);
        gpu_sim.ReleaseMatrix(row_out);
      }
    }

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