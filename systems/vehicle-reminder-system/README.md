# Vehicle Compliance Checker (C++ / C#)

## Project Overview
Vehicle Compliance Checker is a lightweight, data-driven compliance tracking system for logistics and fleet-based SMEs.  
It models vehicle and driver document expiry as a **deterministic time-based state problem**, enabling accurate detection of upcoming non-compliance and preventing missed renewals.

The system uses **C++** as the core validation and time computation engine, and **C#** as a cross-platform presentation layer.  
Vehicle records are ingested from structured CSV data, validated against a strict schema, and evaluated using modern time logic (`std::chrono`).  
Compliance results are exposed via a CLI, JSON contract, and a native GUI for rapid inspection and decision-making.

---

## Core Components
- **C++ Core Engine**
  - CSV parsing and schema validation
  - Deterministic expiry and threshold computation
  - Time-based compliance classification using `std::chrono`

- **C# Application Layer**
  - Cross-platform GUI built with Avalonia
  - Visualisation of compliance status and expiry signals
  - Consumption of JSON outputs from the core engine

---

## Core Features
- Parse and validate vehicle records from CSV data
- Detect schema violations and incomplete entries
- Deterministic time-based expiry evaluation
- Configurable pre-expiry warning windows (e.g. 30 days)
- Machine-readable JSON interface for cross-language integration
- Clear separation between computation logic and I/O / UI layers

---

## Data Model (CSV Schema)
| Field | Description |
|------|-------------|
| Company Name | Fleet owner or operating company |
| Vehicle ID | Plate number or internal identifier |
| Driver License Expiry | Driver license expiry date |
| Permit Expiry | Operating permit expiry date |
| Inspection Expiry | Annual inspection expiry date |
| Last Reminder | Timestamp of last reminder |
| Processed | 0 = pending, 1 = resolved |

---

## Project Structure
- core/   — C++ validation and time logic  
- cli/    — C++ command-line interface  
- gui/    — C# Avalonia GUI  
- data/   — CSV datasets  
- docs/   — Design notes and documentation  
- build/  — Build artifacts (ignored)  
- README.md

---

## Design Focus
- Deterministic, reproducible compliance decisions
- Strong separation between engine and presentation layers
- Language-agnostic JSON contract between C++ and C#
- Performance-oriented core with extensible interfaces

---

## Author
Developed by *Ailurus*  
Engineering practice project focused on time-based modelling, system design, and cross-language architecture.