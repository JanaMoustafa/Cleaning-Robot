#!/usr/bin/env python3
"""
Particle Swarm Optimization (PSO) tuner for Cleaning-Robot.py Q-learning hyperparameters.

This script optimizes continuous hyperparameters and a discrete optimizer choice (mapped from a continuous particle dimension).
It runs Cleaning-Robot.py in headless mode with CONFIG overrides supplied via the Q_CONFIG environment variable and uses
"Headless path length: N" as the objective (minimize).

Note: Keep particle/iteration counts low for quick runs. Increase for better results.
"""
import os
import json
import random
import subprocess
import time
import math
import re
from statistics import mean

SCRIPT = "Cleaning-Robot.py"
PARTICLES = 8
ITERATIONS = 6
REPEATS = 2
TIMEOUT = 90

# Parameter vector layout (indexes)
# 0: alpha [0.01,1.0]
# 1: gamma [0.85,0.999]
# 2: epsilon_decay [0.90,0.995]
# 3: reward_scale [0.5,2.0]
# 4: momentum_beta [0.7,0.99]
# 5: adam_beta1 [0.85,0.95]
# 6: adam_beta2 [0.9,0.9999]
# 7: optimizer_index [0,2] -> {0: 'sgd', 1: 'momentum', 2: 'adam'}

bounds = [
    (0.01, 1.0),    # alpha
    (0.85, 0.999),  # gamma
    (0.90, 0.995),  # epsilon_decay
    (0.5, 2.0),     # reward_scale
    (0.7, 0.99),    # momentum_beta
    (0.85, 0.95),   # adam_beta1
    (0.9, 0.9999),  # adam_beta2
    (0.0, 2.0),     # optimizer_index (continuous)
]

optimizer_map = {0: 'sgd', 1: 'momentum', 2: 'adam'}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def vec_to_config(vec, forced_opt=None):
    # Map particle vector to config dict. If forced_opt provided, use it instead of vec[7].
    if forced_opt is not None:
        opt = forced_opt
    else:
        opt_idx = int(round(clamp(vec[7], bounds[7][0], bounds[7][1])))
        opt_idx = max(0, min(2, opt_idx))
        opt = optimizer_map.get(opt_idx, 'sgd')
    cfg = {
        'alpha': round(vec[0], 4),
        'gamma': round(vec[1], 4),
        'epsilon': 1.0,
        'epsilon_decay': round(vec[2], 4),
        'min_epsilon': 0.01,
        'episodes': 200,
        'max_steps': 200,
        'optimizer': opt,
        'reward_scale': round(vec[3], 4),
    }
    if opt == 'momentum':
        cfg['momentum_beta'] = round(vec[4], 4)
    if opt == 'adam':
        cfg['adam_beta1'] = round(vec[5], 4)
        cfg['adam_beta2'] = round(vec[6], 4)
    if opt == 'sgd':
        # no extra params
        pass 
    return cfg


def evaluate_config(cfg):
    env = os.environ.copy()
    env['Q_CONFIG'] = json.dumps(cfg)
    path_lengths = []
    outs = []
    for r in range(REPEATS):
        try:
            proc = subprocess.run(["python3", SCRIPT, "--nogui"], cwd=os.path.dirname(__file__), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT)
            out = proc.stdout
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ''
        outs.append(out)
        m = re.search(r"Headless path length:\s*(\d+)", out)
        if m:
            path_lengths.append(int(m.group(1)))
        else:
            path_lengths.append(9999)
    return mean(path_lengths), path_lengths, outs


def pso_run():
    # default wrapper: no forced optimizer
    return pso_run_for_optimizer(None)


def pso_run_for_optimizer(forced_opt=None):
    dim = len(bounds)
    # Initialize particles
    particles = []
    velocities = []
    pbest = []
    pbest_scores = []
    for i in range(PARTICLES):
        pos = [random.uniform(lo, hi) for (lo, hi) in bounds]
        vel = [0.0 for _ in range(dim)]
        particles.append(pos)
        velocities.append(vel)
        pbest.append(list(pos))
        pbest_scores.append(float('inf'))

    gbest = None
    gbest_score = float('inf')

    w = 0.5
    c1 = 1.0
    c2 = 1.0

    # Evaluate initial
    print("Evaluating initial swarm...")
    for i, pos in enumerate(particles):
        cfg = vec_to_config(pos, forced_opt=forced_opt)
        score, raw, outs = evaluate_config(cfg)
        pbest_scores[i] = score
        pbest[i] = list(pos)
        print(f" init p{i}: score={score:.2f} cfg={cfg}")
        if score < gbest_score:
            gbest_score = score
            gbest = list(pos)

    print(f"initial gbest score: {gbest_score}")

    for it in range(ITERATIONS):
        print(f"\nPSO Iteration {it+1}/{ITERATIONS}")
        for i in range(PARTICLES):
            # update velocity and position
            r1 = random.random()
            r2 = random.random()
            new_vel = []
            for d in range(dim):
                vel = w * velocities[i][d] + c1 * r1 * (pbest[i][d] - particles[i][d]) + c2 * r2 * (gbest[d] - particles[i][d])
                new_vel.append(vel)
            velocities[i] = new_vel
            # move
            new_pos = [particles[i][d] + velocities[i][d] for d in range(dim)]
            # clamp
            for d in range(dim):
                new_pos[d] = clamp(new_pos[d], bounds[d][0], bounds[d][1])
            particles[i] = new_pos

            cfg = vec_to_config(new_pos, forced_opt=forced_opt)
            score, raw, outs = evaluate_config(cfg)
            print(f" p{i} it{it+1}: score={score:.2f} cfg={cfg}")
            # update pbest
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = list(new_pos)
            # update gbest
            if score < gbest_score:
                gbest_score = score
                gbest = list(new_pos)
                print(f"  new gbest! score={gbest_score}")

    best_cfg = vec_to_config(gbest, forced_opt=forced_opt)
    return gbest_score, best_cfg


if __name__ == '__main__':
    start_total = time.time()
    per_optimizer_results = {}
    for opt in ['sgd', 'momentum', 'adam']:
        print(f"\n=== Running PSO sweep for optimizer: {opt} ===")
        start = time.time()
        score, cfg = pso_run_for_optimizer(forced_opt=opt)
        elapsed = time.time() - start
        print(f"Result for {opt}: score={score} (time {elapsed:.1f}s)")
        per_optimizer_results[opt] = {'score': score, 'cfg': cfg, 'time': elapsed}
        # save per-optimizer best
        with open(f'pso_best_{opt}.json', 'w') as f:
            json.dump({'score': score, 'cfg': cfg}, f, indent=2)

    # Summary
    total_elapsed = time.time() - start_total
    print('\n=== PSO PER-OPTIMIZER SUMMARY ===')
    for opt, res in per_optimizer_results.items():
        print(f"{opt}: score={res['score']}, cfg={json.dumps(res['cfg'])}")

    # Save summary
    with open('pso_best_per_optimizer.json', 'w') as f:
        json.dump(per_optimizer_results, f, indent=2)

    print(f"Saved pso_best_per_optimizer.json; total time elapsed: {total_elapsed:.1f}s")
