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

    // Maintain cumulative K and V stacks across rounds to avoid rebuilding
    static Matrix *k_accum = nullptr;
    static Matrix *v_accum = nullptr;

    // Move only the newly added key/value to Shared Memory if needed
    if (keys[i]->GetPosition() == kInGpuHbm) {
      gpu_sim.MoveMatrixToSharedMem(keys[i]);
    }
    if (values[i]->GetPosition() == kInGpuHbm) {
      gpu_sim.MoveMatrixToSharedMem(values[i]);
    }

    if (i == 0) {
      k_accum = keys[0];
      v_accum = values[0];
    } else {
      Matrix *prev_k = k_accum;
      Matrix *new_k = matrix_memory_allocator.Allocate("k_accum");
      gpu_sim.Concat(k_accum, keys[i], new_k, 0, kInSharedMemory);
      k_accum = new_k;
      // Release previous accumulated k (only if it was an allocated concat, i>=2)
      if (i >= 2) gpu_sim.ReleaseMatrix(prev_k);

      Matrix *prev_v = v_accum;
      Matrix *new_v = matrix_memory_allocator.Allocate("v_accum");
      gpu_sim.Concat(v_accum, values[i], new_v, 0, kInSharedMemory);
      v_accum = new_v;
      if (i >= 2) gpu_sim.ReleaseMatrix(prev_v);
    }

    // Compute K^T using a copy so we don't mutate k_accum for future rounds
    Matrix *k_t = matrix_memory_allocator.Allocate("k_t");
    gpu_sim.Copy(k_accum, k_t, kInSharedMemory);
    gpu_sim.Transpose(k_t, kInSharedMemory);

    // Scores = Q * K^T
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_t, scores);
    // k_t is no longer needed after scores
    gpu_sim.ReleaseMatrix(k_t);

    // Softmax over rows: exp then divide by row sums
    Matrix *scores_exp = matrix_memory_allocator.Allocate("scores_exp");
    gpu_sim.MatExp(scores, scores_exp);
    // scores no longer needed after exp
    gpu_sim.ReleaseMatrix(scores);

    Matrix *softmax_mat = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row_r = matrix_memory_allocator.Allocate("row_r");
      gpu_sim.GetRow(scores_exp, r, row_r, kInSharedMemory);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_r, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_r, row_sum, row_soft);
      // row_r and row_sum no longer needed
      gpu_sim.ReleaseMatrix(row_r);
      gpu_sim.ReleaseMatrix(row_sum);
      if (!softmax_mat) {
        softmax_mat = row_soft; // reuse the first row directly
      } else {
        Matrix *old_softmax = softmax_mat;
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_step");
        gpu_sim.Concat(softmax_mat, row_soft, new_softmax, 0, kInSharedMemory);
        softmax_mat = new_softmax;
        // release temporaries
        gpu_sim.ReleaseMatrix(old_softmax);
        gpu_sim.ReleaseMatrix(row_soft);
      }
    }
    // scores_exp no longer needed after softmax construction
    gpu_sim.ReleaseMatrix(scores_exp);

    // Answer = softmax * V_stack (v_accum)
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax_mat, v_accum, answer);
    // softmax_mat can be released after use
    gpu_sim.ReleaseMatrix(softmax_mat);

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