"""Reviewable evidence, not automatic rejection or recapture verdicts."""

from .contracts import GapItem
from .selection import QUALITY_FLAGS


def build_report(clips, scores, links, edges, graph, revisits, config):
    gaps = []
    for clip in clips:
        rows = [s for s in scores if s.clip_id == clip.clip_id]
        run = []

        def flush(run=run, clip=clip):
            if run and run[-1].t_s - run[0].t_s >= config.scoring.interval_min_s:
                gaps.append(
                    GapItem(
                        clip.clip_id,
                        run[0].t_s,
                        run[-1].t_s,
                        "low_quality_interval",
                        "review",
                        {
                            "flags": sorted(
                                {flag for r in run for flag in r.flags if flag in QUALITY_FLAGS}
                            ),
                            "frames": len(run),
                        },
                    )
                )
            run.clear()

        for row in rows:
            if QUALITY_FLAGS.intersection(row.flags):
                run.append(row)
            else:
                flush()
        flush()
        texture_run = []
        for row in [*rows, None]:
            if row is not None and "low_texture" in row.flags:
                texture_run.append(row)
            else:
                if (
                    texture_run
                    and texture_run[-1].t_s - texture_run[0].t_s >= config.scoring.interval_min_s
                ):
                    gaps.append(
                        GapItem(
                            clip.clip_id,
                            texture_run[0].t_s,
                            texture_run[-1].t_s,
                            "low_texture_interval",
                            "info",
                            {"frames": len(texture_run)},
                        )
                    )
                texture_run.clear()
    for link in links:
        if not link["supported"]:
            gaps.append(
                GapItem(
                    link["clip_id"],
                    link["t_start_s"],
                    link["t_end_s"],
                    "tracking_break" if not link["reliable"] else "weak_link",
                    "review",
                    {k: v for k, v in link.items() if k not in {"clip_id", "t_start_s", "t_end_s"}},
                )
            )
    if len(clips) > 1:
        for clip in clips:
            cross_link = any(
                e.a.split(":")[0] != e.b.split(":")[0]
                and clip.clip_id in {e.a.split(":")[0], e.b.split(":")[0]}
                for e in edges
            )
            if not cross_link:
                gaps.append(
                    GapItem(
                        clip.clip_id,
                        0,
                        clip.duration_s,
                        "unlinked_clip",
                        "review",
                        {"revisits_status": revisits["status"]},
                    )
                )
    return {
        "gaps": [g.to_dict() for g in gaps],
        "revisits": revisits,
        "loop_links": sum(e.kind == "revisit" for e in edges),
        "component_count": len(graph["components"]),
        "timing_gap_info": [
            {"frame_id": s.frame_id, "dt_s": s.dt_s} for s in scores if "timing_gap_info" in s.flags
        ],
        "notes": [
            "Quality flags and missing loop links are evidence for review, not verdicts.",
            "Sequential weights use KLT survival, not verified 3D overlap.",
        ],
    }
