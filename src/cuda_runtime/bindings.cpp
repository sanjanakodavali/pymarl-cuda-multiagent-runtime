#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

void score_actions_cpu(const float* states, float* out_scores,
                       int num_agents, int state_dim, int num_actions);
void score_actions_cuda(const float* d_states, float* d_out_scores,
                        int num_agents, int state_dim, int num_actions);

py::array_t<float> score_actions_cpu_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> states,
    int num_agents, int state_dim, int num_actions) {

    auto buf = states.unchecked<2>(); // [num_agents, state_dim]

    std::vector<float> h_states(num_agents * state_dim);
    std::vector<float> h_scores(num_agents * num_actions);

    for (int i = 0; i < num_agents; ++i)
        for (int d = 0; d < state_dim; ++d)
            h_states[i * state_dim + d] = buf(i, d);

    score_actions_cpu(h_states.data(), h_scores.data(),
                      num_agents, state_dim, num_actions);

    py::array_t<float> out({num_agents, num_actions});
    auto out_buf = out.mutable_unchecked<2>();

    for (int i = 0; i < num_agents; ++i)
        for (int a = 0; a < num_actions; ++a)
            out_buf(i, a) = h_scores[i * num_actions + a];

    return out;
}

py::array_t<float> score_actions_cuda_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> states,
    int num_agents, int state_dim, int num_actions) {

    auto buf = states.unchecked<2>();

    size_t n_state  = (size_t)num_agents * state_dim;
    size_t n_scores = (size_t)num_agents * num_actions;

    std::vector<float> h_states(n_state);
    for (int i = 0; i < num_agents; ++i)
        for (int d = 0; d < state_dim; ++d)
            h_states[i * state_dim + d] = buf(i, d);

    float *d_states = nullptr, *d_scores = nullptr;
    cudaMalloc(&d_states,  n_state  * sizeof(float));
    cudaMalloc(&d_scores, n_scores * sizeof(float));

    cudaMemcpy(d_states, h_states.data(),
               n_state * sizeof(float), cudaMemcpyHostToDevice);

    score_actions_cuda(d_states, d_scores,
                       num_agents, state_dim, num_actions);

    std::vector<float> h_scores(n_scores);
    cudaMemcpy(h_scores.data(), d_scores,
               n_scores * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_states);
    cudaFree(d_scores);

    py::array_t<float> out({num_agents, num_actions});
    auto out_buf = out.mutable_unchecked<2>();

    for (int i = 0; i < num_agents; ++i)
        for (int a = 0; a < num_actions; ++a)
            out_buf(i, a) = h_scores[i * num_actions + a];

    return out;
}

PYBIND11_MODULE(cuda_planner, m) {
    m.def("score_actions_cpu",  &score_actions_cpu_py,  "CPU action scoring");
    m.def("score_actions_cuda", &score_actions_cuda_py, "CUDA action scoring");
}
