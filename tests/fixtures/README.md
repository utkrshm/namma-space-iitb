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
