# LinkedIn — Post 001

Kernel: 44 microseconds.
Program: ~72 milliseconds.

So the math was never the slow part.

I profiled the simplest GPU program — vector add, n ≈ 16.7M — end to end with `rocprofv3` on HIP:

→ hipMalloc ×3 dominated first-touch setup
→ H2D + D2H copies still dwarfed the kernel
→ vector_add itself: ~0.06% of wall clock

Honest check before I'd demo this in an interview:
Is PCIe (and setup) in your story — or are you only admiring the kernel timer?

Your numbers will differ on your GPU. That's the point — re-run and quote *your* trace.

Full write-up (measurement table, snippet, reproduce steps):
https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1

Lab: `tracks/A-gpu-programming/A01.foundations-and-programming-model/`
`make hip && make run-hip` → `make profile-hip`

Series: Parallel Spectrum · Post 001 · Track A
Next: same math, 4.8× faster — how you read memory decides bandwidth (A02).

#GPUProgramming #ParallelComputing #HIP #CUDA #PerformanceEngineering

---
**Posting notes**
- Attach the compiled infographic (`visual.png` or the Figma export) — image posts outperform text-only.
- Medium **friend link** (Share menu on the story) lets non-members read; prefer it if posting to a broad audience.
- Variant: move the Medium URL to the **first comment** if reach matters more than clicks this week.
- Best posting window for engineering audience: Tue–Thu, 9–11 AM your readers' timezone.
