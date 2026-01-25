
content = r"""# NILM Energy Monitor - Project Management Documentation

**Document Version:** Final  
**Project:** NILM Energy Monitor  
**Date:** January 2026  
**Methodology:** Agile Scrum  
**Course:** Team Project (MTS3)  
**Client:** Jeroen De Baets (Energy Lab / Howest)

---

## 1. Methodology Overview

### 1.1 Agile Scrum Framework

The project adopted the **Agile Scrum** methodology, tailored to the intensive "Team Project" module structure. The development was divided into distinct phases: an initial planning phase in November 2025, followed by three intensive development weeks in January 2026.

**Key Scrum Elements Implemented:**
- **Weekly Sprints:** Due to the short duration (3 weeks), sprints were set to 1 week to ensure rapid iteration.
- **Daily Standups:** Short, synchronous meetings (09:00 AM) to align on daily goals and remove blockers.
- **Sprint Reviews:** Conducted at key milestones (End of Week 2 for Interim Demo, End of Week 3 for Final).
- **Retrospectives:** Held weekly to improve team collaboration and technical processes.

### 1.2 Team Structure

The team consisted of 3 members, sharing responsibilities dynamically while maintaining core focus areas:

| Team Member | Core Focus | Responsibilities |
|-------------|------------|------------------|
| **Tommaso Pioda** | Architecture & Integration | Backend (FastAPI), Database (Influx/Supabase), Docker, Deployment |
| **Rodrigo Sousa** | Frontend & UX | React (Vite), Dashboard Design, User Testing, Visualization |
| **Mirko Keller** | AI & Data Science | NILM Model Training (PyTorch), Data Preprocessing, Inference Pipeline |

**Product Owner (Client):** Jeroen De Baets (Energy Lab)  
**Coaches:** Marie Dewitte, Wouter Gevaert, Frederik Waeyaert

---

## 2. Project Timeline

The project followed a strict academic schedule:

| Phase | Dates | Goals |
|-------|-------|-------|
| **Introduction Week** | Nov 3 - Nov 7, 2025 | Research, Architecture Design, Backlog Creation |
| **Holiday Break** | Nov 8 - Jan 4 | Independent Study, Environment Setup |
| **Sprint 1 (Week 1)** | Jan 5 - Jan 9, 2026 | Core ML Models, Data Pipeline, Basic UI |
| **Sprint 2 (Week 2)** | Jan 12 - Jan 16, 2026 | **Interim Demo (20%)**, Full Integration, Real-time Inference |
| **Sprint 3 (Week 3)** | Jan 19 - Jan 23, 2026 | User Testing, Refinement, Documentation, Final Prep |
| **Jury** | Jan 28, 2026 | **Final Presentation (50%)** |

---

## 3. Sprint Planning & Backlogs

### 3.1 Introduction Week (Sprint 0)
**Goal:** Define the problem and solution architecture.
*   **Done:** Defined Seq2Point architecture.
*   **Done:** Selected tech stack (FastAPI, React, InfluxDB, Docker).
*   **Done:** Created initial Trello/GitHub backlog.

### 3.2 Sprint 1: Foundation & MVP (Jan 5-9)
**Goal:** Establishing the data pipeline and training the first model.

| User Story / Task | Status | Owner |
|-------------------|--------|-------|
| [ML] Implement Data Loader for UK-DALE/Refit data | ✅ Done | Mirko |
| [BE] Setup InfluxDB and write/read API wrappers | ✅ Done | Tommaso |
| [FE] Setup React project with Tailwind & Recharts | ✅ Done | Rodrigo |
| [ML] Train baseline CNN Seq2Point model | ✅ Done | Mirko |
| [DevOps] Docker Compose for full stack | ✅ Done | Tommaso |

### 3.3 Sprint 2: Integration & Interim Demo (Jan 12-16)
**Goal:** Connecting Model to Backend and Frontend for a working Demo.
*Milestone:* Interim Presentation (Friday 16/01)

| User Story / Task | Status | Owner |
|-------------------|--------|-------|
| [BE] Create Inference Endpoint (connect API to ML) | ✅ Done | Tommaso/Mirko |
| [FE] Visualize Real-time Energy Graph | ✅ Done | Rodrigo |
| [FE] Implement Appliance Breakdown Chart | ✅ Done | Rodrigo |
| [ML] Optimize Inference speed for real-time | ✅ Done | Mirko |
| [Docs] Prepare Interim Presentation slides | ✅ Done | Team |

**Outcome:** Successful demo showing real-time disaggregation of a synthesized signal.

### 3.4 Sprint 3: Polish & Validation (Jan 19-23)
**Goal:** User testing, error handling, and deployment.

| User Story / Task | Status | Owner |
|-------------------|--------|-------|
| [UX] Conduct User Testing (5 users) | ✅ Done | Rodrigo |
| [FE] Add "Confidence Score" visualization | ✅ Done | Rodrigo |
| [BE] Deploy to Cloudflare & Railway | ✅ Done | Tommaso |
| [ML] Train final models for 5 appliances | ✅ Done | Mirko |
| [Docs] Finalize Functional Analysis & User Manual | ✅ Done | Team |

---

## 4. Burndown Analysis

The burndown chart tracks the remaining effort (in hours) over the 3-week execution phase.

- **Start of Jan (Week 1):** High volume of estimated tasks (approx. 120 hours).
- **End of Week 1:** On track, initial environment setup took slightly longer but ML training was ahead of schedule.
- **Mid-Week 2:** Slight "burndown plateau" identified due to integration challenges between Python Pytorch and FastAPI async loops.
- **End of Week 2 (Interim):** 80% of MVP features completed. Scope adjusted to focus on top 5 appliances instead of 10.
- **End of Week 3:** Successful convergence to 0 remaining critical tasks.

**Velocity:** The team maintained a velocity of approximately **80-90 hours of productive work per week** (combined).

---

## 5. Time Tracking

Total project effort tracking across the 3 execution weeks + introduction.

| Category | Hours (Approx) | % Distribution |
|----------|----------------|----------------|
| **Development (Coding)** | 140h | 50% |
| **ML Training & Experiments** | 60h | 21% |
| **Documentation & Reporting** | 40h | 14% |
| **Meetings (Daily + Reviews)** | 20h | 7% |
| **User Testing & UX** | 20h | 7% |
| **Total** | **280h** | **100%** |

*Average per student:* ~90-95 hours total contribution.

---

## 6. Sprint Retrospectives

### 6.1 Sprint 1 Retrospective (Jan 9)
**Start:**
- Using a shared `types` definition for API responses to avoid FE/BE mismatches.
- Documenting ML model input shapes explicitly.

**Stop:**
- Committing large `.pth` model files to Git (moved to gitignore/external storage).

**Continue:**
- Pair programming on the Inference pipeline (very effective).

### 6.2 Sprint 2 Retrospective (Jan 16 - Post Interim)
**Feedback from Jury:**
- "Good progress on the pipeline, but the UI needs to clearly show *confidence* of predictions."
- "Focus on the business value: why does the user care about specific appliance usage?"

**Action Items:**
1. Added `ConfidenceInterval` UI component.
2. Created a "Cost Estimation" view in the dashboard (Sprint 3).

---

## 7. Conclusion

The project successfully adhered to the aggressive 3-week timeline defined in the "Team Project" brief. By utilizing a rigid Scrum framework with weekly milestones, the team navigated the complexity of integrating deep learning with a web stack. The final result is a deployed, user-tested application that meets the client's goal of "analyzing energy consumption smarter and cheaper."
"""

path = r"c:\Users\Tommaso\Documents\HOWEST\TeamProject\MTS3-MCTE-Team-Project-Energy-G1\docs\deliverables\06_PROJECT_MANAGEMENT.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
