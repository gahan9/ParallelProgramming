"""Solution 3 - the oversubscription cliff, and why your acceptance test missed it.

Same harness as ../exercises/03_oversubscription_check.py with the four TODOs filled
in. Commentary, and the answers in full, are in README.md next to this file.

Run: python 03_oversubscription_check.py
"""

BANNER = ("analytic fabric model, not measured: exact capacity arithmetic on the "
          "as-built uplink count")

HOSTS_PER_LEAF = 32
UPLINKS = 16
HOST_GBS = 50.0
UPLINK_GBS = 50.0

SENDER_LADDER = (1, 8, 16, 17, 32)


def check_rates(host_rate, uplink_rate, local_fraction):
    if host_rate <= 0.0 or uplink_rate <= 0.0:
        raise ValueError(
            f"both rates must be positive, got host {host_rate} and "
            f"uplink {uplink_rate}")
    if not 0.0 <= local_fraction < 1.0:
        raise ValueError(f"local_fraction must be in [0, 1), got {local_fraction}")


def achievable_rate(hosts_per_leaf, uplinks, host_rate, uplink_rate,
                    concurrent_senders, local_fraction=0.0):
    """Per-sender off-leaf rate in GB/s with `concurrent_senders` hosts sending."""
    check_rates(host_rate, uplink_rate, local_fraction)
    if uplinks < 1:
        raise ValueError(f"uplinks must be at least 1, got {uplinks}")
    if concurrent_senders < 1:
        raise ValueError(
            f"concurrent senders must be at least 1, got {concurrent_senders}")
    if concurrent_senders > hosts_per_leaf:
        raise ValueError(
            f"{concurrent_senders} concurrent senders on a leaf with "
            f"{hosts_per_leaf} hosts")
    offered = host_rate * (1.0 - local_fraction)
    return min(offered, uplinks * uplink_rate / concurrent_senders)


def saturation_point(uplinks, host_rate, uplink_rate, local_fraction=0.0):
    """Largest sender count the uplink pool still carries at the offered rate."""
    check_rates(host_rate, uplink_rate, local_fraction)
    if uplinks < 1:
        raise ValueError(f"uplinks must be at least 1, got {uplinks}")
    offered = host_rate * (1.0 - local_fraction)
    return int((uplinks * uplink_rate) // offered)


def implied_bisection_fraction(hosts_per_leaf, uplinks, host_rate, uplink_rate):
    """Fraction of full bisection left for cross-leaf traffic: 1/ratio, capped at 1."""
    check_rates(host_rate, uplink_rate, 0.0)
    if hosts_per_leaf < 1 or uplinks < 1:
        raise ValueError(
            f"hosts and uplinks must be at least 1, got {hosts_per_leaf} hosts and "
            f"{uplinks} uplinks")
    ratio = (hosts_per_leaf * host_rate) / (uplinks * uplink_rate)
    return min(1.0, 1.0 / ratio)


def classify(hosts_per_leaf, uplinks, host_rate, uplink_rate, concurrency,
             local_fraction=0.0):
    """Verdict for one traffic pattern on one leaf."""
    check_rates(host_rate, uplink_rate, local_fraction)
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    ratio = (hosts_per_leaf * host_rate) / (uplinks * uplink_rate)
    if ratio <= 1.0:
        return "non-blocking"
    if concurrency <= saturation_point(uplinks, host_rate, uplink_rate,
                                       local_fraction):
        return "safe for this pattern"
    return "oversubscribed and exposed"


def main():
    print(BANNER)
    host_capacity = HOSTS_PER_LEAF * HOST_GBS
    uplink_capacity = UPLINKS * UPLINK_GBS
    print(f"leaf: {HOSTS_PER_LEAF} hosts x {HOST_GBS:g} GB/s down = "
          f"{host_capacity:g} GB/s, {UPLINKS} uplinks x {UPLINK_GBS:g} GB/s up = "
          f"{uplink_capacity:g} GB/s")
    print(f"oversubscription ratio {host_capacity / uplink_capacity:g}:1\n")

    header = (f"{'senders k':>10}{'uplink demand':>15}{'per-host GB/s':>15}"
              f"{'of line rate':>14}")
    print(header)
    print("-" * len(header))
    for k in SENDER_LADDER:
        rate = achievable_rate(HOSTS_PER_LEAF, UPLINKS, HOST_GBS, UPLINK_GBS, k)
        if rate is None:
            print(f"{k:>10}  not implemented (TODO 1)")
            continue
        print(f"{k:>10}{k * HOST_GBS:>15g}{rate:>15.2f}"
              f"{rate / HOST_GBS * 100:>13.1f}%")
    print("-" * len(header))

    saturation = saturation_point(UPLINKS, HOST_GBS, UPLINK_GBS)
    print(f"\nsaturation point: "
          f"{'not implemented (TODO 2)' if saturation is None else saturation}")
    fraction = implied_bisection_fraction(HOSTS_PER_LEAF, UPLINKS, HOST_GBS, UPLINK_GBS)
    print(f"implied bisection for cross-leaf traffic: "
          f"{'not implemented (TODO 3a)' if fraction is None else f'{fraction * 100:.1f}% of full'}")

    print("\nassertions that must hold. Note what `saturation_point` cannot express: "
          "it has no")
    print("host-count parameter at all, so the cliff cannot move when hosts are added.")
    checks = (
        ("saturation equals the uplink count when rates are equal",
         lambda: saturation_point(16, 50.0, 50.0), lambda v: v == 16),
        ("doubling the uplink rate doubles it",
         lambda: saturation_point(16, 50.0, 100.0), lambda v: v == 32),
        ("half the uplinks halves the saturation point",
         lambda: saturation_point(8, 50.0, 50.0), lambda v: v == 8),
        ("75% locality quadruples it",
         lambda: saturation_point(16, 50.0, 50.0, 0.75), lambda v: v == 64),
        ("a 1:1 leaf is non-blocking",
         lambda: classify(32, 32, 50.0, 50.0, 32), lambda v: v == "non-blocking"),
        ("a 2:1 leaf at full concurrency is exposed",
         lambda: classify(32, 16, 50.0, 50.0, 32),
         lambda v: v == "oversubscribed and exposed"),
        ("the same leaf with 75% locality is safe",
         lambda: classify(32, 16, 50.0, 50.0, 32, 0.75),
         lambda v: v == "safe for this pattern"),
    )
    for label, produce, holds in checks:
        try:
            value = produce()
        except ValueError as exc:
            print(f"  {label:<62} ValueError: {exc}")
            continue
        if value is None or (isinstance(value, tuple) and value[0] is None):
            print(f"  {label:<62} not implemented")
        else:
            print(f"  {label:<62} {'OK' if holds(value) else 'FAILS'}")

    print("\nrejection checks (each must raise ValueError, not return a number):")
    cases = (
        ("0 senders", lambda: achievable_rate(32, 16, 50.0, 50.0, 0)),
        ("33 senders on a 32-host leaf",
         lambda: achievable_rate(32, 16, 50.0, 50.0, 33)),
        ("uplink rate 0", lambda: achievable_rate(32, 16, 50.0, 0.0, 8)),
        ("local_fraction 1.0", lambda: achievable_rate(32, 16, 50.0, 50.0, 8, 1.0)),
    )
    for label, produce in cases:
        try:
            outcome = produce()
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        except ZeroDivisionError:
            outcome = "ZeroDivisionError -- validate instead of dividing"
        else:
            outcome = "no ValueError raised" if outcome is None else f"returned {outcome}"
        print(f"  {label:<32} -> {outcome}")

    print("\nTODO 4 answers are in README.md, section 'Drill 3'.")


if __name__ == "__main__":
    main()

# TODO 4 answers (full reasoning in README.md):
# a) It licenses one claim: a single host can reach line rate off-leaf when nothing
#    else is sending. It says nothing about the ratio, because k = 1 is below the
#    saturation point of every leaf ever built -- a 1:1 leaf and a 4:1 leaf return the
#    same number.
# b) Raise the CONCURRENCY past the uplink count: have all 32 hosts send off-leaf at
#    once and compare the per-host rate against line rate. Message size does not help
#    because the cliff is a capacity limit, not a protocol effect -- one host at 1 GiB
#    still only fills one uplink's worth of the pool.
# c) The radix counts every port as if it were an uplink, so it reports the
#    non-blocking bisection: 2048 link-units where the as-built figure is 1024. Wrong
#    HIGH by exactly the oversubscription ratio, 2x here, which is the direction that
#    gets a design approved.
