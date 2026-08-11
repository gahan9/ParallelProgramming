"""Exercise 3 - the oversubscription cliff, and why your acceptance test missed it.

Oversubscription is the only quantity in this module that a competent team can be
completely wrong about while every benchmark they ran was correct. The arithmetic is
four lines. The trap is that the region below the cliff is exactly where a two-rank
bandwidth test lives, and in that region a 4:1 leaf and a 1:1 leaf are indistinguishable.

Your job:
  TODO 1  implement achievable_rate
  TODO 2  implement saturation_point -- and do NOT let the host count into it
  TODO 3  implement implied_bisection_fraction and classify
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
That gap is the exercise. Worked solution with commentary: ../solutions/README.md

Run: python 03_oversubscription_check.py
"""

BANNER = ("analytic fabric model, not measured: exact capacity arithmetic on the "
          "as-built uplink count")

# Section 2.3's worked leaf, in GB/s: 50 GB/s is 400 Gb/s, so these are the same
# numbers the README states in Gb/s.
HOSTS_PER_LEAF = 32
UPLINKS = 16
HOST_GBS = 50.0
UPLINK_GBS = 50.0

SENDER_LADDER = (1, 8, 16, 17, 32)


def achievable_rate(hosts_per_leaf, uplinks, host_rate, uplink_rate,
                    concurrent_senders, local_fraction=0.0):
    """TODO 1: per-sender off-leaf rate in GB/s, with `concurrent_senders` sending.

        offered = host_rate * (1 - local_fraction)
        pool    = uplinks * uplink_rate
        rate    = min(offered, pool / concurrent_senders)

    Below saturation every sender gets the rate it offered, because the idle uplinks
    absorb it -- that is the whole reason the cliff is invisible from below.

    Raise ValueError with a one-line message when `concurrent_senders` is below 1 or
    above `hosts_per_leaf` (a leaf cannot have more senders than hosts), when either
    rate is non-positive, or when local_fraction is outside [0, 1).
    """
    return None


def saturation_point(uplinks, host_rate, uplink_rate, local_fraction=0.0):
    """TODO 2: largest sender count the uplink pool still carries at the offered rate.

        floor(uplinks * uplink_rate / (host_rate * (1 - local_fraction)))

    Note the signature: `hosts_per_leaf` is deliberately NOT a parameter. Capacity is
    a property of the uplinks, and if your saturation point moves when hosts are added
    to the leaf you have modelled fairness rather than capacity. With equal host and
    uplink rates and no locality the answer must be exactly the uplink count, and the
    first DEGRADED sender count is one more than that.
    """
    return None


def implied_bisection_fraction(hosts_per_leaf, uplinks, host_rate, uplink_rate):
    """TODO 3a: fraction of full bisection left for cross-leaf traffic.

        ratio    = (hosts_per_leaf * host_rate) / (uplinks * uplink_rate)
        fraction = min(1, 1 / ratio)

    Oversubscription and bisection are the same fact at two scales, so a 2:1 leaf
    halves the fabric's bisection for anything crossing leaves. Compute it from the
    as-built uplink count and never from the switch radix -- that substitution is the
    standard way a design document is wrong by exactly 2x.
    """
    return None


def classify(hosts_per_leaf, uplinks, host_rate, uplink_rate, concurrency,
             local_fraction=0.0):
    """TODO 3b: return one of three strings for this pattern on this leaf.

        "non-blocking"                ratio <= 1, nothing to expose
        "safe for this pattern"       oversubscribed, but this concurrency and this
                                      locality stay inside the uplink pool
        "oversubscribed and exposed"  this pattern will drive the pool past saturation

    Three outcomes rather than two, because "oversubscribed" on its own is not a
    verdict: it is a property of the leaf, and whether it costs you anything is a
    property of the traffic.
    """
    return None


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

    print("\nTODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) Your acceptance report shows full line rate between two nodes. State")
    print("     what that licenses you to claim about this leaf, and what it does not.")
    print("  b) Give the smallest change to that test which would actually detect a")
    print("     2:1 leaf, and say why raising the message size is not it.")
    print("  c) A colleague computes the fabric's bisection from the switch radix")
    print("     instead of the uplink count. State the size and the sign of the error.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
