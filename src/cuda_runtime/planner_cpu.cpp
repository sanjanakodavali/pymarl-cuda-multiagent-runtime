#include <vector>

void score_actions_cpu(const float* states, float* out_scores,
                       int num_agents, int state_dim, int num_actions) {
    // Toy scoring: sum(state) * (1 + 0.1 * action_id)
    for (int i = 0; i < num_agents; ++i) {
        const float* s = states + i * state_dim;

        float base = 0.0f;
        for (int d = 0; d < state_dim; ++d) {
            base += s[d];
        }

        for (int a = 0; a < num_actions; ++a) {
            float w = 1.0f + 0.1f * a;
            out_scores[i * num_actions + a] = base * w;
        }
    }
}
