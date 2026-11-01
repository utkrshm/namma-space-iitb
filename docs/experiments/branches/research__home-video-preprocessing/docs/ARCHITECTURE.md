# Planned architecture

The diagram outlines the components of the proposed capture-to-navigation pipeline. Their responsibilities are explained below; project scope and required outcomes are defined in [GOALS.md](GOALS.md).

```text
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

## 1. Capture / SOP

The capture Standard Operating Procedure guides how a person records an indoor space with a smartphone. It covers camera movement, exposure, motion and spatial coverage so that the footage contains usable, overlapping observations of rooms and their connections.

## 2. Video ingestion and preprocessing

This stage decodes the video and extracts frames while preserving their source identity and timing. Quality filtering identifies unusable observations, such as severely blurred frames. Keyframe selection reduces redundant views while retaining useful coverage and connections between parts of the scene. This stage is the current implementation focus.

## 3. 3D perception

The geometric front end estimates spatial information from the selected images. Camera poses describe camera position and orientation; intrinsics describe image projection; depth describes the distance to observed surfaces. Point maps associate image pixels with 3D positions, while tracks associate observations of the same point across views.

VGGT and VGGT-Ω are examples shown in the diagram. The available predictions, and whether point maps or tracks are produced directly or derived separately, depend on the selected method.

## 4. Geometric reconstruction

Reconstruction combines the estimated observations into a coherent scene representation. It aligns and fuses points, removes inconsistent observations, estimates surfaces and generates geometry suitable for downstream use. The visual representation and the geometry used for spatial reasoning serve different purposes.

### Photorealistic representation: 3D Gaussian splatting

Gaussian splatting represents the scene using 3D Gaussian primitives whose appearance is fitted to captured images using estimated cameras. Rendering these primitives supports views from positions other than the original capture viewpoints. Its role is to provide the visual appearance of the digital twin.

### Scene geometry: mesh / TSDF

Explicit geometry represents surfaces and spatial boundaries for collision checking and navigation. A mesh describes surfaces with vertices and faces. A truncated signed distance field (TSDF) stores distances near observed surfaces and can support fusion and mesh extraction. This representation supplies geometric information that a photorealistic rendering alone does not establish.

## 5. Scene understanding

Scene understanding adds meaning to the reconstructed space. Object detections locate items, instance masks distinguish their visible regions, and semantic labels describe their categories. Open-vocabulary embeddings support matching user descriptions to scene content beyond a fixed list of labels.

## 6. 3D semantic map

The semantic map connects object and region descriptions to their spatial locations. It associates observations across views so that an item seen repeatedly can be represented as one scene instance, with a position, extent, category and searchable description or embedding.

## 7. POI / scene database

The scene database organises named destinations such as reception, meeting rooms and printers. It supports tagging, lookup and selection of those destinations in the viewer. Labels may be assigned by a person or assisted by scene understanding; a visual category alone does not establish a venue-specific name such as “Meeting Room 4”.

## 8. Navigation graph

Navigation represents where a visitor can walk and which areas are blocked by walls or furniture. A navigation mesh or graph connects traversable regions and destination locations. A pathfinding method, such as A*, searches this representation to produce a floor-constrained route to a selected POI.

## User-facing applications

### Web walkthrough

The web application presents the visual scene, POI interaction, search and route guidance. First-person exploration works independently of active navigation, using desktop keyboard/mouse controls and suitable mobile controls. Scene geometry supports movement constraints, while the visual representation provides the rendered environment.

### Phone navigation: optional AR

The phone-navigation branch represents the optional live-camera AR experience, distinct from mobile web exploration. Localisation connects the phone's current pose to the reconstructed scene so that destination guidance can be placed in the camera view and updated as the user moves.
