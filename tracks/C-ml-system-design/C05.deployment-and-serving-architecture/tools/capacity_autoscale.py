#!/usr/bin/env python3
"""Seeded simulation of GPU autoscaling against a diurnal traffic pattern.

Standard library only (math, random, argparse, sys). No numpy.

The point of this tool is to make one specific tradeoff measurable: a GPU
replica takes minutes to become useful because it has to load tens of GB of
weights, but traffic can double in seconds. Every scaling policy therefore
buys latency with money or money with latency. The simulation prints both
sides of that trade.

ALL NUMBERS PRINTED BY THIS TOOL ARE SIMULATED. They come from the model
below, not from a measured deployment. The *shape* of the result is the
lesson; the absolute values depend entirely on the parameters you pass.

Three panels
------------
  1. Policy comparison over one simulated day: static-at-peak, reactive
     autoscaling, and reactive autoscaling with a pre-warmed pool.
  2. Cold-start sensitivity: the same reactive policy at several weight-load
     times. This isolates how much of the pain is the cold start itself.
  3. A sparse, bursty workload where scale-to-zero is tempting, showing what
     the first request after an idle gap actually experiences.

Model (deliberately simple, stated so you can argue with it)
------------------------------------------------------------
* Time advances in 60-second ticks.
* Demand is a sinusoidal diurnal curve times multiplicative lognormal noise,
  plus an optional flash spike.
* A ready replica serves `capacity_rps` requests/second. Work that cannot be
  served this tick stays in a queue.
* Admission control: the queue is capped at `shed_after_s` seconds of work.
  Arrivals beyond that are shed (fast 503) rather than queued forever. Real
  systems do this; without it a simulation reports absurd multi-minute
  latencies that no production stack would ever let happen.
* Tick latency = fixed service latency + queue_depth / total_service_rate.
  This is a fluid approximation: it captures queueing collapse, not
  per-request tail behaviour. For true tail latency, load-test the real thing.
* A replica requested at tick t becomes ready at t + cold_start. That delay
  is the whole problem.
* Scale-down waits for a cooldown so the controller does not thrash.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

# 15-second ticks: fine enough that a 30-second cold start is distinguishable
# from a 60-second one, coarse enough that a day is 5760 cheap iterations.
TICK_SECONDS = 15
TICKS_PER_HOUR = 3600 // TICK_SECONDS
TICKS_PER_DAY = 24 * TICKS_PER_HOUR


NO_CAPACITY_MS = 1.0e9      # sentinel: no replica was loaded, so there is no latency to quote


def minutes_to_ticks(minutes):
    return int(round(minutes * 60.0 / TICK_SECONDS))


# --------------------------------------------------------------------------
# Demand generators
# --------------------------------------------------------------------------


def diurnal_demand(seed, base_rps, amplitude, noise, spike_at_minute, spike_factor,
                   spike_minutes=15):
    """One day of per-tick request rates (requests/second)."""
    rng = random.Random(seed)
    trough_tick = minutes_to_ticks(300)     # quietest around 05:00
    spike_start = minutes_to_ticks(spike_at_minute) if spike_at_minute is not None else None
    spike_end = spike_start + minutes_to_ticks(spike_minutes) if spike_start is not None else None

    demand = []
    for t in range(TICKS_PER_DAY):
        phase = 2.0 * math.pi * (t - trough_tick) / TICKS_PER_DAY
        shape = 1.0 + amplitude * math.sin(phase - math.pi / 2.0)
        rps = max(0.5, base_rps * shape * math.exp(rng.gauss(0.0, noise)))
        if spike_start is not None and spike_start <= t < spike_end:
            rps *= spike_factor
        demand.append(rps)
    return demand


def sparse_demand(seed, burst_rps, bursts, burst_minutes):
    """A low-volume internal tool: long idle gaps punctuated by short bursts.

    This is the traffic shape that makes scale-to-zero look irresistible on a
    cost dashboard and indefensible on a latency dashboard.
    """
    rng = random.Random(seed)
    demand = [0.0] * TICKS_PER_DAY
    burst_ticks = minutes_to_ticks(burst_minutes)
    starts = sorted(rng.sample(range(TICKS_PER_HOUR, TICKS_PER_DAY - burst_ticks), bursts))
    for start in starts:
        for t in range(start, start + burst_ticks):
            demand[t] = burst_rps * math.exp(rng.gauss(0.0, 0.15))
    return demand


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------


class StaticPolicy:
    """Fixed replica count. The cost ceiling and the risk floor."""

    def __init__(self, name, replicas):
        self.name = name
        self.replicas = replicas
        self.start_replicas = replicas

    def desired(self, rps, queue, capacity_rps):
        return self.replicas


class QueueDepthAutoscaler:
    """Scale on offered load plus a queue-drain term.

    Deliberately NOT CPU utilization and NOT raw GPU utilization. A GPU
    running one continuous batch reports ~100% busy at almost any request
    rate, so utilization saturates long before the queue does and carries
    almost no information about how far behind you are.
    """

    def __init__(self, name, min_replicas, max_replicas, target_utilization,
                 drain_seconds, warm_pool=0, initial_rps=0.0, capacity_rps=1.0):
        self.name = name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_utilization = target_utilization
        self.drain_seconds = drain_seconds
        self.warm_pool = warm_pool
        self.start_replicas = self.desired(initial_rps, 0.0, capacity_rps)

    def desired(self, rps, queue, capacity_rps):
        effective_rps = rps + queue / self.drain_seconds
        need = effective_rps / (capacity_rps * self.target_utilization)
        return max(self.min_replicas,
                   min(self.max_replicas, int(math.ceil(need)) + self.warm_pool))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def simulate(demand, policy, capacity_rps, cold_start_ticks, scale_down_cooldown,
             slo_ms, service_ms, shed_after_s):
    ready = policy.start_replicas
    warming = []            # ready-at tick for each replica currently loading
    below_target_for = 0

    total = served_total = violating = shed = 0.0
    gpu_ticks = busy_ticks = 0.0
    queue = 0.0
    cold_starts = 0
    latency_samples = []    # (latency_ms, weight in requests)
    subsamples = 5          # sub-tick resolution for latency and SLO accounting

    for t, rps in enumerate(demand):
        newly_ready = [r for r in warming if r <= t]
        ready += len(newly_ready)
        warming = [r for r in warming if r > t]

        arrivals = rps * TICK_SECONDS
        total += arrivals

        service_capacity = ready * capacity_rps * TICK_SECONDS
        queue_cap = max(ready * capacity_rps * shed_after_s, 0.0)

        admitted = min(arrivals, max(0.0, queue_cap + service_capacity - queue))
        shed += arrivals - admitted
        violating += arrivals - admitted        # a shed request is a failed request

        queue_before = queue
        backlog = queue + admitted
        served = min(backlog, service_capacity)
        queue = backlog - served
        served_total += served

        if ready == 0:
            # Nothing is loaded. Every request waits for the cold start.
            latency_samples.append((NO_CAPACITY_MS, arrivals))
        elif admitted > 0:
            # Queue depth moves roughly linearly across the tick, so sample the
            # wait a few times instead of charging every request the end-of-tick
            # value. Without this, SLO accounting is quantised to whole ticks
            # and a warm pool that shaves 20 seconds off the ramp looks free.
            drain = ready * capacity_rps
            wait_start = queue_before / drain
            wait_end = queue / drain
            weight = admitted / subsamples
            for k in range(subsamples):
                frac = (k + 0.5) / subsamples
                wait = wait_start + frac * (wait_end - wait_start)
                latency = service_ms + wait * 1000.0
                latency_samples.append((latency, weight))
                if latency > slo_ms:
                    violating += weight

        gpu_ticks += ready + len(warming)       # a loading replica occupies the GPU too
        busy_ticks += min(served / (capacity_rps * TICK_SECONDS), max(ready, 1))

        want = policy.desired(rps, queue, capacity_rps)
        provisioned = ready + len(warming)
        if want > provisioned:
            for _ in range(want - provisioned):
                warming.append(t + cold_start_ticks)
                cold_starts += 1
            below_target_for = 0
        elif want < ready:
            below_target_for += 1
            if below_target_for >= scale_down_cooldown:
                ready = want
                below_target_for = 0
        else:
            below_target_for = 0

    gpu_hours = gpu_ticks * TICK_SECONDS / 3600.0
    busy_hours = busy_ticks * TICK_SECONDS / 3600.0
    return {
        "policy": policy.name,
        "requests": total,
        "violation_pct": 100.0 * violating / total if total else 0.0,
        "shed_pct": 100.0 * shed / total if total else 0.0,
        "gpu_hours": gpu_hours,
        "idle_gpu_hours": max(0.0, gpu_hours - busy_hours),
        "utilization_pct": 100.0 * busy_hours / gpu_hours if gpu_hours else 0.0,
        "p50_ms": weighted_percentile(latency_samples, 0.50),
        "p99_ms": weighted_percentile(latency_samples, 0.99),
        "p999_ms": weighted_percentile(latency_samples, 0.999),
        "cold_starts": cold_starts,
    }


def weighted_percentile(samples, q):
    """Percentile of tick latency weighted by the requests that experienced it."""
    live = [(v, w) for v, w in samples if w > 0]
    if not live:
        return 0.0
    live.sort()
    target = q * sum(w for _, w in live)
    running = 0.0
    for value, weight in live:
        running += weight
        if running >= target:
            return value
    return live[-1][0]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def fmt_ms(value):
    if value >= NO_CAPACITY_MS:
        return "cold"
    return "%.0f" % value


def print_demand_profile(demand, label):
    peak = max(demand)
    trough = min(demand)
    mean = sum(demand) / len(demand)
    print("%s (simulated, 24 h in %d-second ticks)" % (label, TICK_SECONDS))
    ratio = ("%.1fx" % (peak / trough)) if trough > 0 else "infinite (fully idle periods)"
    print("  mean %.1f rps, peak %.0f rps, trough %.1f rps, peak/trough %s"
          % (mean, peak, trough, ratio))
    levels = " .:-=+*#%@"
    hourly = [sum(demand[h * TICKS_PER_HOUR:(h + 1) * TICKS_PER_HOUR]) / TICKS_PER_HOUR
              for h in range(24)]
    bar = "".join(levels[min(len(levels) - 1, int(len(levels) * v / (peak * 1.001)))]
                  for v in hourly)
    print("  hour  000000000011111111112222")
    print("        012345678901234567890123")
    print("  load [%s]" % bar)


def print_table(rows, cost_per_gpu_hour):
    header = ("%-32s %9s %8s %8s %9s %10s %7s %8s %9s"
              % ("policy", "SLO viol", "shed", "p50 ms", "p99 ms", "p99.9 ms",
                 "util", "GPU-h", "$/day"))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("%-32s %8.2f%% %7.2f%% %8s %9s %10s %6.0f%% %8.1f %9.0f"
              % (r["policy"], r["violation_pct"], r["shed_pct"], fmt_ms(r["p50_ms"]),
                 fmt_ms(r["p99_ms"]), fmt_ms(r["p999_ms"]), r["utilization_pct"],
                 r["gpu_hours"], r["gpu_hours"] * cost_per_gpu_hour))


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def panel_policies(args, demand, cold_start_ticks, peak_replicas):
    print()
    print("PANEL 1 -- policy comparison, cold start %ds" % args.cold_start_s)
    print("=" * 92)
    initial = demand[0]
    policies = [
        StaticPolicy("static @ peak (%d replicas)" % peak_replicas, peak_replicas),
        QueueDepthAutoscaler("reactive autoscale (floor 1)", 1, peak_replicas + 8,
                             args.target_utilization, args.drain_seconds,
                             initial_rps=initial, capacity_rps=args.capacity_rps),
        QueueDepthAutoscaler("reactive + warm pool 2", 1, peak_replicas + 8,
                             args.target_utilization, args.drain_seconds, warm_pool=2,
                             initial_rps=initial, capacity_rps=args.capacity_rps),
        QueueDepthAutoscaler("reactive + warm pool 5", 1, peak_replicas + 12,
                             args.target_utilization, args.drain_seconds, warm_pool=5,
                             initial_rps=initial, capacity_rps=args.capacity_rps),
        QueueDepthAutoscaler("reactive + warm pool 10", 1, peak_replicas + 16,
                             args.target_utilization, args.drain_seconds, warm_pool=10,
                             initial_rps=initial, capacity_rps=args.capacity_rps),
    ]
    rows = [simulate(demand, p, args.capacity_rps, cold_start_ticks, args.scale_down_cooldown,
                     args.slo_ms, args.service_ms, args.shed_after_s) for p in policies]
    print_table(rows, args.cost_per_gpu_hour)
    return rows


def panel_cold_start(args, demand, peak_replicas):
    print()
    print("PANEL 2 -- cold-start sensitivity for the reactive autoscaler (floor 1, no warm pool)")
    print("=" * 92)
    print("  Same policy, same traffic, same seed. Only the weight-load time changes.")
    print()
    rows = []
    for cold_start_s in args.cold_start_sweep:
        ticks = max(1, int(round(cold_start_s / TICK_SECONDS)))
        policy = QueueDepthAutoscaler("cold start %4ds" % cold_start_s, 1, peak_replicas + 8,
                                      args.target_utilization, args.drain_seconds,
                                      initial_rps=demand[0], capacity_rps=args.capacity_rps)
        rows.append(simulate(demand, policy, args.capacity_rps, ticks, args.scale_down_cooldown,
                             args.slo_ms, args.service_ms, args.shed_after_s))
    print_table(rows, args.cost_per_gpu_hour)
    return rows


def panel_scale_to_zero(args):
    print()
    print("PANEL 3 -- scale-to-zero on a sparse, bursty workload")
    print("=" * 92)
    demand = sparse_demand(args.seed, args.burst_rps, args.bursts, args.burst_minutes)
    print_demand_profile(demand, "  sparse internal-tool traffic")
    print()
    ticks = max(1, int(round(args.cold_start_s / TICK_SECONDS)))
    peak_replicas = max(1, int(math.ceil(max(demand) / (args.capacity_rps * args.target_utilization))))
    policies = [
        QueueDepthAutoscaler("scale-to-zero (floor 0)", 0, peak_replicas + 4,
                             args.target_utilization, args.drain_seconds,
                             initial_rps=0.0, capacity_rps=args.capacity_rps),
        QueueDepthAutoscaler("floor 1 (always one replica)", 1, peak_replicas + 4,
                             args.target_utilization, args.drain_seconds,
                             initial_rps=0.0, capacity_rps=args.capacity_rps),
        StaticPolicy("static @ peak (%d replicas)" % peak_replicas, peak_replicas),
    ]
    rows = [simulate(demand, p, args.capacity_rps, ticks, args.scale_down_cooldown,
                     args.slo_ms, args.service_ms, args.shed_after_s) for p in policies]
    print_table(rows, args.cost_per_gpu_hour)
    return rows


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(args):
    if args.cold_start_s < 0:
        raise ValueError("--cold-start-s cannot be negative")
    if args.capacity_rps <= 0:
        raise ValueError("--capacity-rps must be positive")
    if not 0.0 < args.target_utilization <= 1.0:
        raise ValueError("--target-utilization must be in (0, 1]")

    demand = diurnal_demand(args.seed, args.base_rps, args.amplitude, args.noise,
                            args.spike_at_minute, args.spike_factor)
    cold_start_ticks = max(1, int(round(args.cold_start_s / TICK_SECONDS)))
    peak_replicas = int(math.ceil(max(demand) / (args.capacity_rps * args.target_utilization)))

    print("=" * 100)
    print("GPU AUTOSCALING SIMULATION -- every figure below is SIMULATED, not measured")
    print("=" * 100)
    print("  seed %d | %.0f rps per replica | cold start %ds (%d ticks) | SLO %.0f ms"
          % (args.seed, args.capacity_rps, args.cold_start_s, cold_start_ticks, args.slo_ms))
    print("  service latency at empty queue %.0f ms | shed beyond %.0fs of queued work | $%.2f/GPU-h"
          % (args.service_ms, args.shed_after_s, args.cost_per_gpu_hour))
    print()
    print_demand_profile(demand, "  primary product traffic")
    print("  flash spike: x%.1f for 15 minutes starting at minute %d of the day"
          % (args.spike_factor, args.spike_at_minute))
    print("  replicas required at peak: %d" % peak_replicas)

    policy_rows = panel_policies(args, demand, cold_start_ticks, peak_replicas)
    panel_cold_start(args, demand, peak_replicas)
    zero_rows = panel_scale_to_zero(args)

    baseline = policy_rows[0]
    reactive = policy_rows[1]
    print()
    print("HOW TO READ THIS")
    print("=" * 100)
    print("  1. Static-at-peak is the honest upper bound on cost and lower bound on risk:")
    print("     $%.0f/day, %.2f%% violations, %.0f%% utilization. You are paying for the"
          % (baseline["gpu_hours"] * args.cost_per_gpu_hour, baseline["violation_pct"],
             baseline["utilization_pct"]))
    print("     evening peak all night long.")
    print("  2. Reactive autoscaling saves $%.0f/day (%.0f%%) and buys %+.2f pp of SLO"
          % ((baseline["gpu_hours"] - reactive["gpu_hours"]) * args.cost_per_gpu_hour,
             100.0 * (1 - reactive["gpu_hours"] / baseline["gpu_hours"]),
             reactive["violation_pct"] - baseline["violation_pct"]))
    print("     violations. Whether that trade is acceptable is a product decision, not an")
    print("     infrastructure one -- take it to the person who owns the SLO.")
    print("  3. The warm pool is priced insurance. It shows up directly in GPU-hours and")
    print("     directly reduces p99 during the ramp and the flash spike. Present it that")
    print("     way to a budget owner rather than as a technical detail.")
    print("  4. Panel 2 is the strategic result: shrinking the cold start moves the whole")
    print("     frontier. Local NVMe weight cache, faster checkpoint formats, and keeping a")
    print("     warm image beat any amount of controller tuning.")
    print("  5. Panel 3: scale-to-zero costs $%.0f/day less than a single always-on replica,"
          % ((zero_rows[1]["gpu_hours"] - zero_rows[0]["gpu_hours"]) * args.cost_per_gpu_hour))
    print("     and takes SLO violations from %.2f%% to %.2f%%. If your SLO is anywhere near"
          % (zero_rows[1]["violation_pct"], zero_rows[0]["violation_pct"]))
    print("     the cold start, scale-to-zero is not a cost optimization -- it is an outage")
    print("     you scheduled in advance, once per idle gap.")


def build_parser():
    p = argparse.ArgumentParser(
        description="Seeded diurnal-traffic autoscaling simulation for GPU model serving.")
    p.add_argument("--seed", type=int, default=20240729)
    p.add_argument("--base-rps", type=float, default=120.0, help="mean request rate")
    p.add_argument("--amplitude", type=float, default=0.75, help="diurnal swing, 0-1")
    p.add_argument("--noise", type=float, default=0.12,
                   help="lognormal sigma of minute-to-minute noise")
    p.add_argument("--spike-at-minute", type=int, default=780,
                   help="minute of the day at which a 15-minute flash spike starts")
    p.add_argument("--spike-factor", type=float, default=2.5)
    p.add_argument("--capacity-rps", type=float, default=18.0, help="requests/second per replica")
    p.add_argument("--cold-start-s", type=float, default=180.0,
                   help="seconds to load weights and become ready")
    p.add_argument("--cold-start-sweep", type=float, nargs="+", default=[15, 30, 60, 180, 300, 600])
    p.add_argument("--target-utilization", type=float, default=0.70)
    p.add_argument("--drain-seconds", type=float, default=30.0,
                   help="how fast the controller wants to drain an existing queue")
    p.add_argument("--scale-down-cooldown", type=int, default=40,
                   help="ticks below target before shrinking")
    p.add_argument("--shed-after-s", type=float, default=5.0,
                   help="seconds of queued work beyond which requests are shed")
    p.add_argument("--slo-ms", type=float, default=1500.0)
    p.add_argument("--service-ms", type=float, default=400.0)
    p.add_argument("--cost-per-gpu-hour", type=float, default=3.00)
    p.add_argument("--burst-rps", type=float, default=11.0, help="panel 3: rate during a burst")
    p.add_argument("--bursts", type=int, default=12, help="panel 3: bursts per day")
    p.add_argument("--burst-minutes", type=float, default=5.0, help="panel 3: minutes per burst")
    return p


def main(argv=None):
    if sys.version_info < (3, 8):
        sys.stderr.write("ERROR: this tool needs Python 3.8 or newer.\n")
        return 2
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except ValueError as exc:
        sys.stderr.write("ERROR: %s\n" % exc)
        sys.stderr.write("Run with -h for usage.\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
