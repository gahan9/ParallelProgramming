# C07 Exercises

Three drills. Do them **before** reading [`../solutions/`](../solutions/) — a design you have
struggled with for 45 minutes teaches you something a design you have read does not.

| # | File | Clock | Skill |
|---|------|------:|-------|
| 1 | [`01_design_video_moderation.md`](01_design_video_moderation.md) | 45 min | Apply the framework to a fresh prompt with a human-in-the-loop and an asymmetric fairness cost |
| 2 | [`02_design_semantic_search.md`](02_design_semantic_search.md) | 45 min | Hybrid retrieval, reranking, freshness — the bridge between the two case studies and the base of any RAG system |
| 3 | [`03_review_a_design.md`](03_review_a_design.md) | 30 min | Review a plausible but flawed design document and find five serious problems |

Do **3 last**. Reviewing is a different muscle from designing, and it is closer to what a
senior engineer actually spends the week doing.

---

## How to run these

**Set a timer and honour it.** The constraint is the exercise. Producing a beautiful document
in three hours trains the wrong skill; producing a defensible one in 45 minutes trains the
right one.

**Write it down, do not just think it.** Use the ten-step framework from
[`../README.md` §2](../README.md#2-first-principles) as literal headings. Filling every heading,
even with one line, is how you find the section you would otherwise have skipped.

**Do the arithmetic on paper first, then check it.** Both calculators accept the parameters
these drills need:

```bash
python ../tools/llm_sizing.py --help
python ../tools/funnel_budget.py --help
```

The habit being trained is *estimate first, verify second*. If you reach for the tool before
you have a rough number in your head, you are using it as a crutch rather than a check.

**Score yourself** against the rubric in
[`../README.md` §9.4](../README.md#94-self-review-rubric) before reading the solution, then
score again after. The gap between the two scores is the actual lesson.

---

## What "done" looks like

A finished drill has all ten framework sections present, every quantitative claim traced to a
labelled assumption, at least one latency budget whose hops sum to less than the SLO, and an
explicit list of what you left out. It does **not** need to be right — the solutions
deliberately point out places where a different choice would also have been defensible.

If you find yourself writing "this should be fast enough," stop and put a number there. That
sentence is the single most reliable marker of a design that has not been thought through.
