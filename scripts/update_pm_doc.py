
content = r"""# NILM Energy Monitor – Project Management Documentation

**Document Version:** Final  
**Project:** NILM Energy Monitor  
**Date:** January 2026  
**Methodology:** Agile (Scrum)  

---

## 1. Methodology Overview

### 1.1 Chosen Methodology: Scrum

For the NILM Energy Monitor project, the Scrum framework was adopted as the primary Agile methodology. Scrum proved particularly suitable due to the short academic timeline and the experimental nature of machine learning development. The use of fixed-length sprints allowed the team to align development work with academic deadlines, while regular feedback cycles enabled continuous validation of both technical and functional requirements. Furthermore, Scrum’s flexibility supported iterative refinement of the NILM models as new insights emerged during experimentation, prioritizing working software over excessive documentation.

### 1.2 Team Structure

The project team consisted of three members: **Tommaso Pioda**, **Mirko Keller**, and **Rodrigo Sousa**. Responsibilities were distributed collaboratively following Scrum principles, without rigid role separation. 

- **Frontend & UX**: Focused on React components, styling, and user interaction.
- **Backend & API**: Focused on FastAPI endpoints, database integration, and security.
- **ML & Data**: Focused on model training, the inference pipeline, and data preprocessing.

Scrum responsibilities such as backlog prioritization, facilitation of meetings, and coordination were shared across the team, ensuring collective ownership of the project.

### 1.3 Sprint Configuration

The project was executed over a total duration of **4 weeks**, structured into **two sprints**, each lasting **2 weeks**. 

- **Daily Standups**: 15 minutes, synchronous, used to align and identify blockers.
- **Sprint Reviews**: Conducted at the end of each sprint to demonstrate progress.
- **Retrospectives**: Focused on process improvement after each review.

### 1.4 Tools Used

Several tools were employed to support Agile execution:

| Tool | Purpose |
|------|---------|
| **GitHub Projects & Issues** | Backlog management, task tracking, and sprint planning |
| **Slack** | Day-to-day team communication |
| **VS Code Live Share** | Pair programming sessions |
| **Miro** | Retrospective boards for feedback and improvement actions |

### 1.5 Definition of Done

A user story was considered complete only when:
1. The implemented feature compiled without errors.
2. Relevant unit tests were written and successfully passed.
3. The code had been reviewed by at least one other team member.
4. Documentation was updated where applicable.
5. Verification logic (acceptance criteria) was satisfied.
6. Final acceptance was given collectively by the team.

---

## 2. Sprint Planning

### 2.1 Sprint Overview

Given the reduced project duration, the work was organized into two main sprints:

- **Sprint 1 (Weeks 1-2)**: focused on project setup, data ingestion, and initial machine learning experimentation.
- **Sprint 2 (Weeks 3-4)**: emphasized frontend development, API integration, inference pipelines, testing, and documentation.

Across both sprints, a total of approximately **45 story points** were completed, reflecting a high delivery rate within the limited timeframe.

### 2.2 Sprint 1: Setup, Data Pipeline, and ML Foundations

The primary objective of the first sprint was to establish a solid technical foundation. 

**Key Deliverables:**
- Repository structure set up with Docker Compose.
- Frontend (React/Vite) and Backend (FastAPI) scaffolded.
- Data ingestion pipeline implemented using InfluxDB.
- Initial preprocessing steps defined for NILM training.
- Preliminary ML models (CNN Seq2Point) trained and evaluated.

Although some tasks required more time than initially estimated—particularly model tuning—the sprint successfully delivered its core objectives.

### 2.3 Sprint 2: Frontend, API Integration, and Inference

The second sprint focused on delivering a complete, user-facing application.

**Key Deliverables:**
- Core frontend components (dashboards, charts, appliance lists).
- Backend APIs fully integrated with authentication.
- Real-time inference pipeline serving NILM predictions.
- User testing sessions and bug fixing.
- Comprehensive technical and user documentation.

---

## 3. Sprint Backlog Example

An example backlog from the second sprint included user stories related to authentication, data visualization, and secure API access. Each story was estimated in story points, assigned to a team member, and tracked through GitHub Issues. Most stories were completed within the sprint, with only minor UI-related enhancements deferred to a future backlog.

### 3.1 Task Breakdown Example

**User Story:** "As a user, I can see energy readings for my building"

This task illustrates the value of iterative estimation refinement. The initial estimate was 13 hours, but actual effort was 14.5 hours due to additional backend optimization.

| Task | Estimate (hrs) | Actual (hrs) | Status |
|------|----------------|--------------|--------|
| Design API endpoint specification | 1 | 1 | ✅ Done |
| Implement `/analytics/readings` endpoint | 3 | 4 | ✅ Done |
| Add InfluxDB query logic | 2 | 3 | ✅ Done |
| Create frontend hook `useEnergyData` | 2 | 2 | ✅ Done |
| Integrate chart components | 2 | 2 | ✅ Done |
| Write unit tests | 2 | 1.5 | ✅ Done |
| Code review and merge | 1 | 1 | ✅ Done |
| **TOTAL** | **13h** | **14.5h** | |

---

## 4. Burndown Analysis

Sprint burndown analysis showed a generally consistent completion rate across both sprints. Minor slowdowns occurred when unexpected technical complexity arose, particularly during authentication and model integration. However, these delays were mitigated through pair programming and focused debugging sessions. 

Overall, the project maintained a stable velocity and concluded with a small number of non-critical backlog items intentionally postponed.

---

## 5. Time Tracking

Across the four-week project, the team invested approximately **240 total hours**, averaging around **20 hours per week per team member**. The majority of time was spent on development activities, followed by research and experimentation related to machine learning models.

### 5.1 Time Distribution

| Category | Approx. Hours | Percentage |
|----------|---------------|------------|
| Development (Coding) | 120 | 50% |
| Research & ML Experimentation | 48 | 20% |
| Testing & User Feedback | 36 | 15% |
| Documentation | 24 | 10% |
| Planning & Meetings | 12 | 5% |
| **Total** | **240** | **100%** |

---

## 6. Sprint Retrospectives

All retrospectives followed the **Start–Stop–Continue** format.

### 6.1 Key Insights from Retrospectives

**Sprint 1 Retrospective (Mid-Project)**
- **Start:** Documenting hyperparameters in logs; Sharing model training progress in daily standups.
- **Stop:** Working in isolation on experimental code without pushing implementation details.
- **Continue:** Using Jupyter notebooks for early exploration.

**Sprint 2 Retrospective (Project Conclusion)**
- **Start:** Recording user testing sessions (with consent); Prioritizing issues by frequency.
- **Stop:** Making assumptions about user behavior; Delaying documentation until the very end.
- **Continue:** Pair programming for complex backend integration; Clear API contracts.

### 6.2 Application of Feedback
Retrospectives highlighted the importance of clearly defined API contracts and early usability testing. As a result, the team shifted to a "Docs First" approach for the API in Sprint 2, which significantly reduced integration errors.

---

## 7. Lessons Learned

### 7.1 Technical Lessons
- **Architecture:** Simpler NILM architectures (Seq2Point) can outperform complex ones (Seq2Seq, Transformer) when properly tuned for specific appliances.
- **Loss Functions:** Weighted loss functions are essential for handling the class imbalance inherent in appliance usage data (mostly OFF states).
- **Environment:** Early containerization (Docker) significantly reduced "works on my machine" issues.

### 7.2 Process Lessons
- **Sprints:** Two-week sprints provided good rhythm even within a short project.
- **User Testing:** Provided valuable insights that challenged initial design assumptions; ideally, this should have started even earlier.
- **Collaboration:** Pair programming was highly effective for complex tasks like the inference pipeline.

### 7.3 Risk Management

| Risk | Impact | Mitigation Strategy | Outcome |
|------|--------|---------------------|---------|
| ML model accuracy insufficient | High | Test multiple architectures (CNN, Transformer) | Achieved acceptable accuracy with CNN |
| Data pipeline latency | Medium | Implement caching layer via Redis/Postgres | Performant real-time updates |
| Scope creep | Medium | Strict adherence to story points & MVP goals | Critical features delivered on time |

### 7.4 Final Summary
The NILM Energy Monitor project was completed over 4 weeks using the Scrum methodology, resulting in a fully functional web application for energy disaggregation. Through iterative development, regular retrospectives, and close collaboration, the team achieved its academic and technical objectives while maintaining a strong focus on working software and continuous improvement.
"""

path = r"c:\Users\Tommaso\Documents\HOWEST\TeamProject\MTS3-MCTE-Team-Project-Energy-G1\docs\deliverables\06_PROJECT_MANAGEMENT.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
