import time
import numpy as np
import cuda_planner
import numpy as np

def run_once(fn, runs=20):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return np.array(times)

def benchmark(num_agents=4096, state_dim=64, num_actions=16, runs=20):
    states = np.random.randn(num_agents, state_dim).astype(np.float32)

    # warmup
    cuda_planner.score_actions_cpu(states, num_agents, state_dim, num_actions)
    cuda_planner.score_actions_cuda(states, num_agents, state_dim, num_actions)

    cpu_times = run_once(
        lambda: cuda_planner.score_actions_cpu(states, num_agents, state_dim, num_actions),
        runs=runs,
    )
    gpu_times = run_once(
        lambda: cuda_planner.score_actions_cuda(states, num_agents, state_dim, num_actions),
        runs=runs,
    )

    def stats(name, ts):
        print(
            f"{name}: mean={ts.mean():.3f} ms, "
            f"p50={np.percentile(ts,50):.3f} ms, "
            f"p95={np.percentile(ts,95):.3f} ms"
        )

    stats("CPU", cpu_times)
    stats("CUDA", gpu_times)

    actions = num_agents * num_actions
    print(f"CPU actions/sec  ~ {actions / (cpu_times.mean()/1000):.1f}")
    print(f"CUDA actions/sec ~ {actions / (gpu_times.mean()/1000):.1f}")

if __name__ == "__main__":
    benchmark()
