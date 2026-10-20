# Preprocessing fixture

`room_short.mp4` is the first five seconds of `videos/room_m_block.mp4`,
re-encoded at 640×360 (coded) with original VFR timing. Display rotation is
−90°; upright dimensions are 360×640. It contains no audio, GPS or copied
container metadata. This is a curated real capture, not a generated scene.

Reproduce (the intermediate file may live under `/tmp/opencode`):

```sh
ffmpeg -noautorotate -i videos/room_m_block.mp4 -t 5 -map 0:v:0 -an -vf scale=640:360 -fps_mode passthrough -c:v libx264 -crf 28 -preset fast -map_metadata -1 /tmp/opencode/room_short_encoded.mp4
ffmpeg -i /tmp/opencode/room_short_encoded.mp4 -map 0:v:0 -c copy -map_metadata -1 -metadata:s:v:0 rotate=-90 tests/fixtures/room_short.mp4
```

Tests verify rotation and the absence of location metadata using this small
fixture, not the full videos.

## Ceiling tracking regression

`coorg_ceiling.mp4` is Coorg BnB at 72–80 s: a real slow, sharp pan of a mostly
blank ceiling/wall. Coded size is 640×360, display rotation −90°. The source has
no intentional cut in this interval. Metadata is removed before remuxing rotation:

```sh
ffmpeg -noautorotate -ss 72 -i videos/coorg_bnb.mp4 -t 8 -map 0:v:0 -an -vf scale=640:360 -fps_mode passthrough -enc_time_base 1:90000 -video_track_timescale 90000 -c:v libx264 -crf 18 -preset fast -map_metadata -1 /tmp/opencode/coorg_ceiling_encoded.mp4
ffmpeg -i /tmp/opencode/coorg_ceiling_encoded.mp4 -map 0:v:0 -c copy -video_track_timescale 90000 -map_metadata -1 -metadata:s:v:0 rotate=-90 tests/fixtures/coorg_ceiling.mp4
```

The explicit encoder time base avoids quantizing VFR presentation times onto a
1/60-second lattice. Tests verify monotonic timestamps and no location tags.

## User-labelled blur/texture observations

`coorg_labels/*.png` are metadata-free upright 360×640 greyscale observations
decoded by the real pipeline from `videos/coorg_bnb.mp4`. Each is the closest PTS
to the user's requested time. The source time base is 1/90000 and start PTS is zero.
OpenCV PNG export copies no video metadata.

| File | Requested seconds | Actual seconds | Source PTS | User label |
| --- | ---: | ---: | ---: | --- |
| `coorg_5s.png` | 5 | 4.997111 | 449740 | Sharp |
| `coorg_12_5s.png` | 12.5 | 12.492778 | 1124350 | Sharp, low texture |
| `coorg_25s.png` | 25 | 25.002322 | 2250209 | Sharp |
| `coorg_49_5s.png` | 49.5 | 49.504922 | 4455443 | Motion blur |
| `coorg_103_5s.png` | 103.5 | 103.507433 | 9315669 | Sharp, low texture |
| `coorg_140s.png` | 140 | 140.003056 | 12600275 | Sharp |
| `coorg_172s.png` | 172 | 172.001422 | 15480128 | Sharp, low texture |
| `coorg_204s.png` | 204 | 203.999878 | 18359989 | Sharp, low texture |

The 49.5 s observation's per-frame re-blur loss is about 0.31 at this resolution;
sharp-door observations overlap that range. Its single-frame blur assertion is
an explicit strict expected failure rather than a threshold fitted to eight images.
Temporal corroboration can still flag it, without requiring measured flow. The
sharp 5 s observation also receives an informational `low_texture` annotation:
that means insufficient evidence for blur, not poor quality or exclusion.
