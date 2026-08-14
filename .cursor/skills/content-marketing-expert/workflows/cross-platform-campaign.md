<!-- SPDX-License-Identifier: MIT -->

# Cross-Platform Campaign Workflows

Six end-to-end walkthroughs. Each assumes the ContentBrief from step 1 of the
skill exists, and each ends at the memory-shot question.

## 1. Blog-first, then Reel and Short

Use when a written piece already carries the research.

```
1. Delegate the draft to content-research-writer; return here with the finished post.
2. SEO package: title variants, meta description, H2 questions, alt text, internal links.
3. Mine the post for three claims that stand alone. Each becomes one short-form piece.
4. Per claim: write the hook trio, a 5-6 beat script, and a visual cue per beat.
5. Generate one cover per piece via the asset routing tree; wire paths into the shot lists.
6. Cut a 4-post X thread from the post's structure; hand it to twitter-algorithm-optimizer.
7. Ensemble review across all surfaces at once so the promise stays identical.
8. Deliver, then ask about the memory shot.
```

The trap here is publishing the same sentence on four surfaces. Keep the claim
constant and rewrite the hook per platform.

## 2. Reel-first, then blog expansion and thread

Use when the video exists and the written surfaces are downstream.

```
1. Transcribe the reel (local speech-to-text preferred; see ../references/asset-routing.md).
2. Turn the transcript into an outline: the reel's beats become H2 sections.
3. Expand each beat with the detail that would not fit in 30 seconds.
4. SEO pass on the expanded post; the reel's spoken hook usually seeds the title.
5. Pull the strongest three lines from the transcript into a thread.
6. Reuse the reel's cover as the blog hero if the aspect ratio allows; otherwise
   regenerate at wide aspect using the stored prompt.
7. Ensemble review, deliver, ask about the memory shot.
```

## 3. Generate a thumbnail, then overlay it on a Short

```
1. Read a sibling asset from the destination folder; match style and palette.
2. Generate the cover at 9:16 using a stylized-still prompt (see ../references/generative-models.md).
3. If the local generator is unavailable, fall back per the routing tree and, if
   nothing is reachable, write the image spec plus prompt file and continue.
4. Overlay onto the video: name the position, scale, start time, duration, opacity.
   File-based editors write to a new output path; the original is untouched.
5. Record the asset path, prompt, seed, and the timecodes it occupies.
6. Verify the overlay clears the platform safe areas before delivering.
```

## 4. Batch assets for a carousel or a sectioned reel

```
1. List the sections first. Each section gets one asset and one descriptive filename.
2. Generate sequentially, holding style, palette, and aspect constant across the set.
   Pin the seed where the backend supports it so a re-run reproduces the set.
3. Import or place in section order; scale to fill the frame.
4. Build the slide manifest or the shot list with each path and its position.
5. Spot-check the set side by side — batch generation drifts in style by the last item.
```

Style drift across a set is the common failure. Re-describing the palette
identically in every prompt is cheaper than regenerating the outliers.

## 5. Offline campaign pack

Use when there are no API keys, no network, or no configured generators. This is
a complete deliverable, not a degraded one.

```
1. Full ContentBrief and the five-pass loop, unchanged — none of it needs a backend.
2. Scripts and shot lists in markdown with timecodes and visual cues per beat.
3. Asset specs instead of assets: one prompt file per asset, with size, style,
   palette, and destination recorded.
4. SEO package complete; mark every unverified number TBD-verify.
5. An offline variant where relevant: print or slide copy with a QR code or a
   written contact step instead of a swipe-up.
6. Ensemble review runs normally.
7. Hand over a run-list: exactly which prompts to execute when a backend is available.
```

## 6. SEO refresh on an existing piece

```
1. Read the existing post and its current metadata. Note what already ranks.
2. Re-run the SEO pass against current search suggestions and audience questions.
   Do not invent volumes; mark anything unverified.
3. Rewrite the title and meta description only if the new version is clearly better;
   churning metadata on a ranking page costs more than it gains.
4. Add the sections the piece is missing, phrased as the questions readers ask.
5. Refresh alt text and internal links.
6. Cut two new short-form hooks from the updated angles.
7. Ensemble review, deliver a diff-style summary of what changed and why.
```

## Common closing steps

Every workflow ends the same way:

1. List every file written and the surface it feeds.
2. Report the ensemble scores and the equilibrium notes.
3. Name anything left unverified.
4. Ask once about the memory shot.
