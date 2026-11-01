# Project goals

## Purpose

Build an end-to-end spatial engine for the Namma Space challenge at IIT Bombay Techfest 2026–27. Convert captures of real indoor spaces into photorealistic, interactive 3D digital twins that people can explore, search, and navigate through a web browser.

The intended experience makes indoor spaces understandable without requiring visitors to interpret static floor plans. A person should be able to find a room, zone, or item and see a walkable route around physical obstacles.

## Desired outcomes

- **Practical capture:** A person without technical expertise can capture an indoor space using a smartphone, guided by a team-defined Standard Operating Procedure (SOP).
- **Faithful reconstruction:** Captured data produces a reliable, photorealistic 3D representation without manual 3D modelling.
- **Remote exploration:** Visitors can open a link and freely explore the reconstructed space in first person through a zero-install web application on desktop and mobile browsers. This is the team's primary delivery target.
- **Discoverable destinations:** Named Points of Interest (POIs), such as reception areas, meeting rooms, and product zones, can be tagged, selected, and searched.
- **Walkable navigation:** Selecting a destination provides visual guidance along an optimal walking path that stays on the floor and avoids walls, furniture, and other physical obstacles.
- **Transfer to a new venue:** The capture-to-deployment pipeline can be applied to a previously unseen indoor location within the competition's on-site processing window.

### Optional stretch outcome

Provide on-site augmented reality (AR) navigation through a phone's live camera view, with localization and directional overlays updating as the user walks. The challenge describes this as its longer-term vision, but explicitly classifies AR navigation as a bonus; desktop navigation is mandatory.

## Scope

### Required capabilities

- A documented capture SOP suitable for realistic indoor data collection, covering sufficient space coverage for reconstruction.
- Reconstruction of a real indoor environment from captured data. Smartphone video is the repository's planned starting input; the challenge also permits photos, floor plans, and sensor data.
- Fluid first-person roaming and looking around using standard desktop controls, such as WASD and mouse input. Exploration must work independently of active navigation.
- POI tagging and clickable rooms or zones that reveal their assigned names.
- Spatial search for destinations, including the ability to jump to a matching location in the viewer.
- Active desktop path planning, obstacle-aware path visualization, and guidance to a selected POI.
- Access to exploration, POI search, and navigation through both desktop and mobile web interfaces, with controls suited to each device.
- A working web deployment and supporting evidence demonstrating the complete capture, reconstruction, exploration, search, and navigation experience.

### Demonstration environments

Initial validation uses the team's home or a similarly accessible indoor space. The final demonstration, if the team qualifies, uses a newly assigned venue at IIT Bombay rather than relying solely on a prepared scene.

Office floors, retail stores, mall sections, and event spaces illustrate the intended application domain. They do not establish a guaranteed supported venue size.

Target environments have stationary contents and an unchanged layout during capture and navigation. Navigation is limited to a single floor.

## Success conditions

### Product experience

The required experience succeeds when:

1. A real captured indoor space becomes a structurally coherent, visually faithful 3D environment without manual 3D modelling.
2. A visitor can access the viewer through a working link on desktop or mobile and move freely through the scene without starting navigation.
3. Clicking a labelled room or zone reveals the correct name, and searching for a represented destination leads to the corresponding location.
4. Selecting a reachable POI produces visual navigation along a floor-constrained route around real physical obstacles.
5. The same overall approach supports capture, reconstruction, POI annotation, and deployment for a newly assigned venue within the allotted on-site timeframe.

These conditions reflect the challenge's emphasis on capture practicality, reconstruction fidelity, fluid browser exploration, accurate spatial search, and non-colliding pathfinding. They are intended outcomes, not claims of existing capability.

### Competition deliverables

| Stage | Required evidence of success |
| --- | --- |
| Round 1: reconstruction and roaming | A captured home or similarly accessible space; a technical abstract PDF of at most five pages covering architecture, pipeline, SOP, and approach; a document linking to a working free-roaming web viewer; a 3–5 minute demonstration video; and a repository link or folder with reconstruction setup instructions and sample data in a test directory. |
| Round 2: POIs and navigation | An updated repository branch containing POI and navigation functionality; a live web application demonstrating POI interaction, spatial search, and walkable floor paths; and a 3–5 minute demonstration video showing those capabilities. |
| Round 3: on-site capture and finale | Capture and processing of the assigned venue, reconstruction, POI annotation, and full platform deployment during Day 1; a 10-minute live pitch and platform demonstration and a 5-minute technical Q&A on Day 2. |

The finale evaluates capture execution, reconstruction quality, browser experience, spatial search and path calculation, and presentation equally at 20% each. On-site AR navigation carries a separate 10% bonus.

## Non-goals

- Replacing the captured environment with a manually modelled scene.
- Delivering only a static model or fixed-view tour in place of free first-person exploration.
- Treating a visual shortcut through walls or furniture as a valid walking route.
- Making mobile AR a prerequisite for completion of the mandatory desktop and mobile web experience.
- Multi-floor routing, navigation around moving obstacles, or additional venue-specific navigation cases.
- Guaranteeing reconstruction from arbitrary videos or reliable navigation in every possible indoor environment; the challenge calls for a defined capture SOP and demonstrated results.

## Technical and operational constraints

- Capture must remain practical for users with a smartphone. The challenge permits additional capture inputs and hardware or sensors under the team's SOP; it does not mandate a specific reconstruction technology.
- Third-party tools and open-source libraries are permitted. The submission must be the team's original work, comply with relevant licences, credit reused tools in the technical abstract, and respect third-party intellectual property.
- Online submission folders must be viewable by anyone with the link. Supplementary links and technical materials must be presented in a single organized master PDF or Google Doc, rather than as loose links or unorganized files.
- Competition participation requires registration and a Team ID. Teams may have at most four students with valid institutional identification and may include members from different institutes.
- The final on-site result must be ready within the organizers' allotted capture and processing timeframe. The supplied PDF does not specify the number of processing hours.

### Published competition dates

The supplied challenge PDF lists the following dates. Organizer updates may supersede them.

| Milestone | Published date |
| --- | --- |
| Registration closes | 15 October 2026 |
| Round 1 submission | 31 October 2026 |
| Round 1 results | 10 November 2026 |
| Round 2 submission | 30 November 2026 |
| Finalist announcement | 7 December 2026 |
| On-site capture and processing | 16 December 2026 |
| Live pitch and technical Q&A | 17 December 2026 |

## Known unknowns and decisions requiring input

- Whether the team will pursue the optional mobile AR outcome alongside the mandatory desktop and mobile web application.
- The exact assigned final venue, its extent and access conditions, and the available on-site processing window.
- How “optimal” walking paths will be assessed; the PDF requires optimal, floor-constrained, obstacle-avoiding routes without defining a path-cost metric.

## Sources and interpretation

- [Official Namma Space challenge brief](Namma%20Space.pdf): purpose and required capabilities on pages 1–3; round deliverables and evaluation on pages 3–4; competition rules and submission constraints on page 5; timeline on pages 5–6.
- [Repository instructions](../AGENTS.md): identifies this repository as a Namma Space submission and establishes smartphone-captured video to digital twin with navigation as its purpose.
- [Planned architecture](ARCHITECTURE.md): identifies smartphone RGB video as the planned capture input and includes web walkthrough and phone navigation outcomes.
- [Repository overview](../README.md): summarises the digital-twin and web-navigation goal, with live-camera AR as an optional extension.

This document follows the challenge brief: desktop navigation is required and mobile AR is optional. The challenge's allowance for multiple capture inputs does not imply that every input modality must be supported.

Team scope clarifications additionally establish desktop and mobile web access as mandatory, limit navigation to single-floor environments with stationary contents, and exclude additional venue-specific navigation cases.
