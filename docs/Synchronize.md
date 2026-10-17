# Dubbing, video and subtitle alignment

Basic translation and third-person narration both use `SpeedRate` in
`videotrans/task/_rate.py`. First-person, line-rewrite and adaptive modes have
been removed. Old saved selections of removed modes load as basic translation.

## Timing policy

The TTS speaking-rate setting applies when synthesizing speech. Automatic audio
acceleration is a separate operation on the generated waveform. Its shared
maximum defaults to **1.25x (+25%)**, for every voice. A saved legacy default of
50x migrates to 1.25x. The advanced setting accepts multipliers from 1 to 2.
These are mechanical limits, not guarantees of intelligibility.

For each cue, measure generated audio and allow it to occupy the following
subtitle gap, stopping at the next cue start (or video end). Preserve any unused
part of the gap. If this interval is insufficient, accelerate within the audio
limit and measure again. If it still does not fit and video slowdown is enabled,
extend the video interval to the measured duration. There is no video slowdown
ratio ceiling; timestamp validity, overlap and output duration checks remain.

| Audio acceleration | Video slowdown | Behavior |
| --- | --- | --- |
| On | On | Accelerate first; extend the corresponding video interval for the remainder. |
| On | Off | Accelerate within the limit; report an error if the cue still cannot fit. |
| Off | On | Keep speech speed; extend the video interval to fit it. |
| Off | Off | Preserve legacy concatenation behavior; long speech may shift later cues. |

The last row is intentionally not a synchronization guarantee. Audio-only jobs
also use the shared limit; they cannot fall back to changing video.

## Media processing and verification

- Keep leading silence, individual cue intervals, gaps and the video tail.
- Generate processed 48 kHz stereo PCM from original TTS files; never overwrite
  cached TTS or use a duration cut to discard excess synthesized speech.
- Apply `setpts` to video intervals that need more time. When rebuilding video,
  quantize output pieces to whole frames at a common frame rate. Source cuts
  include the frame containing a sub-frame boundary; fractional frame padding
  can add up to one frame per interval. Video slowdown does not interpolate motion.
- Verify segment and final frame counts. Supply explicit frame-derived durations
  to the concat demuxer, using cumulative microsecond rounding so MP4 container
  rounding cannot accumulate across clips. Measure the final video duration.
  Rebuild cue timestamps
  from the verified timeline, not from accumulated estimates.
- Concatenate speech samples and silence to the final video length. Verify each
  speech clip fits its interval and verify final PCM sample count before publishing.
- Stop on missing audio, invalid/overlapping cue timing, encoder failure,
  unacceptable duration mismatch. Video slowdown has no ratio ceiling; extreme stretching can look nearly frozen.
  Never silently skip failed segments. No output is published on validation failure.
- Save `sync-*/fit_report.json` with per-cue speed/duration and the video time map.

Original vocal and separated BGM tracks are transformed piece by piece using
that same map before mixing, preserving their relation to visual events.
A separately selected music file remains a continuous added music bed and follows
its existing loop setting. Original exported source media are unchanged.

Basic translation retains one source cue per target cue. Third-person narration
uses short phrases and new cue timing inside each scene, with no hard deletion
of over-budget text. It does not imply word-for-word or lip synchronization.
Bilingual embedding by cue index is disabled for third-person narration because
the narration cue count differs from the source.

## Verification

`tests/test_rate_pipeline.py` exercises real FFmpeg processing, the reported
36.62s/24.91s overflow, audible tail markers, a visual motion marker, fractional
frame rates, source-track timing, gaps, failure handling and the speed ceiling.
`tests/test_shared_narration_pipeline.py` checks mode migration, shared alignment
routing and narration content retention. These tests do not validate translation
accuracy or prove that every voice remains intelligible at the configured speed.


## Long-video source separation

Source separation now reads 60-second audio chunks with 2 seconds of context.
Only each chunk's central source interval is retained, so context never adds
time to the output. End padding protects the final samples from model frame
rounding. Vocals and background must both have the exact input sample count.

Completed chunks are stored under `tmp/separation_cache`, keyed by input audio
content, model files and chunk parameters. Recreated task directories can reuse
these checkpoints. This cache consumes disk space and can be removed when no
job needs to resume; do not remove it during an active separation.

Subprocess monitoring follows the PID assigned to each task. A replacement
worker does not count as completion of a crashed task. Separation times out
after 15 minutes without a chunk progress update; other subprocess jobs have
a two-hour inactivity limit. Cancellation terminates only the assigned worker.
App shutdown terminates pools rather than waiting forever for lost results.
These changes take effect after restarting the application.
