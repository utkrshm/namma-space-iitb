```
                        SMARTPHONE
                            │
                            │ RGB video
                            ▼
                  ┌─────────────────────┐
                  │ 1. CAPTURE / SOP    │
                  │                     │
                  │ Camera trajectory   │
                  │ Exposure / motion   │
                  │ Coverage            │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 2. VIDEO INGESTION  │
                  │                     │
                  │ Decode video        │
                  │ Frame extraction    │
                  │ Quality filtering   │
                  │ Keyframe selection  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 3. 3D PERCEPTION    │
                  │                     │
                  │ VGGT / VGGT-Ω      │
                  │                     │
                  │ • camera poses     │
                  │ • intrinsics       │
                  │ • depth            │
                  │ • point maps       │
                  │ • tracks           │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼────────────────┐
              │              │                │
              ▼              ▼                ▼
          Camera poses      Depth         Point cloud
              │              │                │
              │              │                │
              └──────────────┼────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ 4. GEOMETRIC        │
                  │    RECONSTRUCTION    │
                  │                     │
                  │ Point fusion        │
                  │ Outlier removal     │
                  │ Surface estimation  │
                  │ Mesh generation     │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          Dense visual geometry      Scene geometry
                 │                       │
                 ▼                       ▼
            3D Gaussian Splatting    Mesh / TSDF
                 │                       │
                 ▼                       │
          Photorealistic scene          │
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 5. SCENE            │
                  │    UNDERSTANDING     │
                  │                     │
                  │ Object detection   │
                  │ Instance masks     │
                  │ Semantic labels    │
                  │ Open-vocabulary    │
                  │ embeddings         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 6. 3D SEMANTIC MAP  │
                  │                     │
                  │ Objects ↔ 3D       │
                  │ geometry           │
                  │                     │
                  │ "chair_12"         │
                  │ position           │
                  │ extent             │
                  │ category           │
                  │ embedding          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 7. POI / SCENE      │
                  │    DATABASE          │
                  │                     │
                  │ Meeting Room 4     │
                  │ Sofa               │
                  │ Reception          │
                  │ Printer            │
                  │ Conference Room    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 8. NAVIGATION GRAPH │
                  │                     │
                  │ Walkable geometry  │
                  │ Obstacle geometry  │
                  │ NavMesh / graph    │
                  │                     │
                  │ A* / shortest path │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              WEB WALKTHROUGH    PHONE NAVIGATION
                    │                 │
                    ▼                 ▼
              3DGS renderer      Localization
                    │                 │
                    ▼                 ▼
              First-person       Destination
                roaming          guidance
    ```
