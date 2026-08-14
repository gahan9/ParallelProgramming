<!-- SPDX-License-Identifier: MIT -->

# Generative Prompt Tuning

Behavioral guidance for writing prompts, not an API wrapper. It applies whichever
backend is live, and it still applies when none is — a prompt file written to
this standard is executable by a human or a later session.

Three prompt families cover almost every marketing asset. Pick by what the asset
must do, not by which product is available.

| Family | Produces | Backend requirement |
|--------|----------|---------------------|
| Motion-first | Short looping clips with deliberate camera and subject movement | Text-to-video, 2-6s, loop support |
| Cinematic | Longer B-roll with narrative pacing and depth | Text-to-video, 8s+, seed pinning |
| Stylized still | Characterful single frames, covers, overlays | Text-to-image, 9:16 and 1:1 |

## Motion-first prompts

Built for Reels and Shorts B-roll: 2-6 seconds, one visible movement, loopable.

Order the prompt as subject, action, camera, environment, lighting, style,
duration:

```
A single ceramic coffee cup on a wooden desk, steam curling upward,
slow push-in on the cup, morning kitchen with soft window light,
warm neutral palette, shallow depth of field, 4 seconds, seamless loop
```

Rules that matter:

- **One movement per clip.** A camera move and a subject action competing in four
  seconds reads as noise.
- **Name the camera move explicitly** — push-in, orbit, static, handheld drift.
  Left unsaid, the model picks something arbitrary.
- **State the duration and whether it loops.** Loop-friendly clips cut together
  without a visible seam.
- **Avoid legible text.** Motion models render text unreliably; add text in the
  edit as an overlay.

## Cinematic prompts

Built for hero sequences and longer B-roll where pacing carries meaning.

Add shot grammar the motion-first family does not need: shot size, lens
character, grade, and the emotional beat.

```
Wide establishing shot of an empty studio at dawn, dust in the light beams,
anamorphic lens character, slow lateral dolly left, cool blue shadows with
warm highlight rim, contemplative and still, 8 seconds
```

Rules that matter:

- **Specify shot size** (wide, medium, close) so a sequence cuts together.
- **Keep the grade consistent** across every clip in one sequence; describe it in
  the same words each time.
- **Match cut points to the script beats** rather than generating a clip and
  writing copy to fit it.
- **Continuity is manual.** These models do not remember a subject between
  generations. Re-describe the subject identically in every prompt of a sequence,
  and pin a seed when the backend supports it.

## Stylized still prompts

Built for covers, thumbnails, overlays, and carousel slides.

```
Flat vector illustration of a hand holding a phone showing a rising chart,
bold two-color palette of deep indigo and warm coral, thick clean outlines,
centered composition on a light neutral background, no text
```

Rules that matter:

- **Subject, style, composition, palette, background** — all five, always.
- **Quote any text that must appear** and keep it to a handful of words. If the
  copy is critical or long, generate the background clean and overlay the text in
  the edit where it stays editable.
- **Match the destination.** Read a sibling asset first and mirror its style and
  palette; delegate to `theme-factory` when the project has a defined theme.
- **Do not assume transparency.** Many models cannot produce it. Generate on a
  flat background and cut it out in a separate step if a matte is needed.

## Negative guidance

Where the backend supports a negative prompt, the reliably useful entries for
marketing assets are: extra fingers, distorted hands, garbled text, watermark,
logo, low resolution, oversaturated, generic stock-photo look.

## Reproducibility

Every generated asset gets its prompt recorded next to it — in a prompt file or
in the Assets table. A campaign that cannot regenerate its own assets is not
resumable, and the memory shot depends on those prompts to continue the work in a
later session.

## Iteration

Change one variable at a time. Changing the subject, the lighting, and the style
together tells you nothing about which change produced the result. Keep the seed
fixed while iterating on wording, then vary the seed once the wording is right.
