#include <cuda_runtime.h>

__global__
void score_actions_kernel(const float* __restrict__ states,
                          float* __restrict__ out_scores,
                          int num_agents, int state_dim, int num_actions) {
    int agent_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int action_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (agent_id >= num_agents || action_id >= num_actions) return;

    const float* s = states + agent_id * state_dim;

    float base = 0.0f;
    for (int d = 0; d < state_dim; ++d) {
        base += s[d];
    }

    float w = 1.0f + 0.1f * action_id;
    out_scores[agent_id * num_actions + action_id] = base * w;
}

extern "C"
void score_actions_cuda(const float* d_states, float* d_out_scores,
                        int num_agents, int state_dim, int num_actions) {
    dim3 block(16, 16);
    dim3 grid((num_agents + block.x - 1) / block.x,
              (num_actions + block.y - 1) / block.y);

    score_actions_kernel<<<grid, block>>>(d_states, d_out_scores,
                                          num_agents, state_dim, num_actions);
}
