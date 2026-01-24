# NILM Energy Monitor - Project Management Documentation

**Document Version:** Final  
**Project:** NILM Energy Monitor  
**Date:** January 2026  
**Methodology:** Agile (Scrum)  

---

## Table of Contents

1. [Methodology Overview](#1-methodology-overview)
2. [Sprint Planning](#2-sprint-planning)
3. [Sprint Backlog Example](#3-sprint-backlog-example)
4. [Burndown Analysis](#4-burndown-analysis)
5. [Time Tracking](#5-time-tracking)
6. [Sprint Retrospectives](#6-sprint-retrospectives)
7. [Lessons Learned](#7-lessons-learned)

---

## 1. Methodology Overview

### 1.1 Chosen Methodology: Scrum

We adopted **Scrum** as our primary Agile framework for the NILM Energy Monitor project. This choice was driven by:

- Clear sprint boundaries for academic milestones
- Regular feedback cycles with stakeholders
- Flexibility to adapt to discoveries during ML model development
- Emphasis on working software over documentation

### 1.2 Team Structure

| Role | Responsibility |
|------|----------------|
| **Product Owner** | Define priorities, accept deliverables |
| **Scrum Master** | Facilitate ceremonies, remove blockers |
| **Development Team** | Frontend, Backend, ML development |

**Team Size:** 4 members

### 1.3 Sprint Configuration

| Parameter | Value |
|-----------|-------|
| **Sprint Duration** | 2 weeks |
| **Total Sprints** | 8 sprints (16 weeks) |
| **Daily Standups** | 15 minutes, synchronous |
| **Sprint Review** | End of sprint, demo to stakeholders |
| **Sprint Retrospective** | Internal, after review |

### 1.4 Tools Used

| Tool | Purpose |
|------|---------|
| **GitHub Projects** | Backlog management, Kanban board |
| **GitHub Issues** | User stories, tasks, bugs |
| **Slack** | Team communication |
| **VS Code Live Share** | Pair programming |
| **Miro** | Retrospective boards |

### 1.5 Definition of Done

A user story is considered "Done" when:
- [ ] Code is implemented and compiles without errors
- [ ] Unit tests written and passing
- [ ] Code reviewed by at least one team member
- [ ] Documentation updated (if applicable)
- [ ] Feature merged to development branch
- [ ] Product Owner accepts the feature

---

## 2. Sprint Planning

### 2.1 Sprint Overview

| Sprint | Dates | Theme | Velocity (SP) |
|--------|-------|-------|---------------|
| Sprint 1 | Sep 9-22 | Project Setup & Architecture | 18 |
| Sprint 2 | Sep 23 - Oct 6 | Data Pipeline & Database | 24 |
| Sprint 3 | Oct 7-20 | ML Model Development | 21 |
| Sprint 4 | Oct 21 - Nov 3 | Frontend Core Components | 26 |
| Sprint 5 | Nov 4-17 | API Integration & Auth | 28 |
| Sprint 6 | Nov 18 - Dec 1 | Inference Pipeline | 24 |
| Sprint 7 | Dec 2-15 | User Testing & Iteration | 22 |
| Sprint 8 | Dec 16 - Jan 15 | Polish & Documentation | 20 |

**Total Story Points Completed:** 183

### 2.2 Sprint 1: Project Setup

**Goal:** Establish project foundation and architecture

**Key Deliverables:**
- Repository structure created
- Docker Compose configuration
- Frontend scaffolding (React + Vite)
- Backend scaffolding (FastAPI)
- CI/CD pipeline (GitHub Actions)

**Story Points Planned:** 20  
**Story Points Completed:** 18  
**Velocity:** 18

### 2.3 Sprint 2: Data Pipeline

**Goal:** Implement data ingestion and storage

**Key Deliverables:**
- InfluxDB integration
- Data preprocessing pipeline
- Model-ready data format
- Supabase schema design

**Story Points Planned:** 26  
**Story Points Completed:** 24  
**Velocity:** 24

### 2.4 Sprint 3: ML Model Development

**Goal:** Train initial NILM disaggregation models

**Key Deliverables:**
- CNN Seq2Point model
- Transformer architecture
- Training pipeline
- Model evaluation metrics

**Story Points Planned:** 25  
**Story Points Completed:** 21  
**Velocity:** 21

**Notes:** Model training took longer than estimated due to hyperparameter tuning.

### 2.5 Sprint 4: Frontend Core

**Goal:** Build main user interface components

**Key Deliverables:**
- Dashboard page
- Chart components (Recharts)
- Appliance list view
- Date range selector
- Responsive layout

**Story Points Planned:** 28  
**Story Points Completed:** 26  
**Velocity:** 26

### 2.6 Sprint 5: API & Authentication

**Goal:** Connect frontend to backend with auth

**Key Deliverables:**
- API endpoints implementation
- Supabase authentication
- JWT validation in backend
- Protected routes
- Building/appliance CRUD

**Story Points Planned:** 30  
**Story Points Completed:** 28  
**Velocity:** 28

### 2.7 Sprint 6: Inference Pipeline

**Goal:** Real-time prediction flow

**Key Deliverables:**
- Model loading and caching
- Inference endpoint
- Prediction storage
- Confidence scoring
- Demo mode implementation

**Story Points Planned:** 26  
**Story Points Completed:** 24  
**Velocity:** 24

### 2.8 Sprint 7: User Testing

**Goal:** Validate usability and iterate

**Key Deliverables:**
- User testing sessions (12 participants)
- Bug fixes from testing
- UI/UX improvements
- Onboarding flow
- Export functionality

**Story Points Planned:** 24  
**Story Points Completed:** 22  
**Velocity:** 22

### 2.9 Sprint 8: Polish & Docs

**Goal:** Final polish and documentation

**Key Deliverables:**
- Performance optimization
- Error handling improvements
- User manual
- Technical documentation
- Presentation materials

**Story Points Planned:** 22  
**Story Points Completed:** 20  
**Velocity:** 20

---

## 3. Sprint Backlog Example

### 3.1 Sprint 5 Backlog (API & Authentication)

| ID | User Story | Priority | Estimate (SP) | Status | Assignee |
|----|------------|----------|---------------|--------|----------|
| US-23 | As a user, I can log in with email/password | High | 5 | ✅ Done | Member A |
| US-24 | As a user, I am redirected to login if not authenticated | High | 3 | ✅ Done | Member A |
| US-25 | As a user, I can view my buildings after login | High | 5 | ✅ Done | Member B |
| US-26 | As a user, I can see energy readings for my building | High | 8 | ✅ Done | Member B |
| US-27 | As an admin, I can add new buildings | Medium | 5 | ✅ Done | Member C |
| US-28 | As a user, my session persists across page refreshes | Medium | 3 | ✅ Done | Member A |
| US-29 | As a developer, API validates JWT signatures | High | 5 | ✅ Done | Member D |
| US-30 | As a user, I see helpful error messages on API failure | Low | 2 | ⏳ Partial | Member C |

**Sprint Goal:** Users can authenticate and view their building data securely.

**Total Planned:** 30 SP  
**Total Completed:** 28 SP  

### 3.2 Task Breakdown Example: US-26

**User Story:** As a user, I can see energy readings for my building

| Task | Estimate (hrs) | Actual (hrs) | Status |
|------|----------------|--------------|--------|
| Design API endpoint specification | 1 | 1 | ✅ |
| Implement `/analytics/readings` endpoint | 3 | 4 | ✅ |
| Add InfluxDB query logic | 2 | 3 | ✅ |
| Create frontend hook `useEnergyData` | 2 | 2 | ✅ |
| Integrate chart with data | 2 | 2 | ✅ |
| Write unit tests | 2 | 1.5 | ✅ |
| Code review and merge | 1 | 1 | ✅ |

**Total Estimate:** 13 hours  
**Total Actual:** 14.5 hours

---

## 4. Burndown Analysis

### 4.1 Sprint 5 Burndown Chart (Described)

The burndown chart for Sprint 5 illustrates the following pattern:

**Ideal Line:** Straight diagonal from 30 SP (Day 1) to 0 SP (Day 14)

**Actual Line:**
- **Days 1-3:** On track, completed 8 SP (authentication stories)
- **Days 4-6:** Slight delay, API endpoint complexity discovered
- **Days 7-10:** Accelerated completion after blockers removed
- **Days 11-14:** Final 6 SP completed, ended at 2 SP remaining

**Key Observations:**
- Mid-sprint dip due to unexpected JWT validation issues
- Team recovered velocity after pair programming session
- 2 SP (US-30) carried to next sprint

### 4.2 Overall Project Burndown

**Total Product Backlog:** 220 Story Points

| Sprint | Remaining SP (Start) | Completed | Remaining SP (End) |
|--------|---------------------|-----------|-------------------|
| 1 | 220 | 18 | 202 |
| 2 | 202 | 24 | 178 |
| 3 | 178 | 21 | 157 |
| 4 | 157 | 26 | 131 |
| 5 | 131 | 28 | 103 |
| 6 | 103 | 24 | 79 |
| 7 | 79 | 22 | 57 |
| 8 | 57 | 20 | 37 |

**Final Status:** 37 SP remaining (non-critical, moved to future backlog)

### 4.3 Velocity Trend

```
Sprint 1: ██████████████████ 18 SP
Sprint 2: ████████████████████████ 24 SP
Sprint 3: █████████████████████ 21 SP
Sprint 4: ██████████████████████████ 26 SP
Sprint 5: ████████████████████████████ 28 SP (Peak)
Sprint 6: ████████████████████████ 24 SP
Sprint 7: ██████████████████████ 22 SP
Sprint 8: ████████████████████ 20 SP

Average Velocity: 22.9 SP/sprint
```

---

## 5. Time Tracking

### 5.1 Team Time Summary

| Team Member | Role Focus | Total Hours | Avg Hours/Week |
|-------------|------------|-------------|----------------|
| Member A | Frontend Lead | 248 | 15.5 |
| Member B | Backend Lead | 264 | 16.5 |
| Member C | ML Engineer | 232 | 14.5 |
| Member D | Full-Stack | 240 | 15.0 |

**Total Team Hours:** 984 hours  
**Project Duration:** 16 weeks

### 5.2 Time by Activity Category

| Category | Hours | Percentage |
|----------|-------|------------|
| Development (Coding) | 492 | 50% |
| Research & Learning | 148 | 15% |
| Testing & Debugging | 128 | 13% |
| Documentation | 89 | 9% |
| Meetings & Communication | 79 | 8% |
| Code Review | 48 | 5% |

### 5.3 Weekly Time Log (Sprint 5 Example)

| Week | Mon | Tue | Wed | Thu | Fri | Sat | Sun | Total |
|------|-----|-----|-----|-----|-----|-----|-----|-------|
| Week 9 | 4h | 5h | 4h | 5h | 3h | 2h | 0h | 23h |
| Week 10 | 5h | 4h | 6h | 4h | 4h | 3h | 2h | 28h |

**Sprint 5 Total:** 51 hours (average per member: 12.75h)

### 5.4 Time per Feature Area

| Feature Area | Hours | % of Total |
|--------------|-------|------------|
| Dashboard & Visualization | 156 | 15.9% |
| ML Model Training | 142 | 14.4% |
| API Development | 128 | 13.0% |
| Authentication | 86 | 8.7% |
| Data Pipeline | 78 | 7.9% |
| Infrastructure (Docker/CI) | 72 | 7.3% |
| Testing | 68 | 6.9% |
| User Testing & Iteration | 64 | 6.5% |
| Documentation | 89 | 9.0% |
| Meetings | 79 | 8.0% |
| Other | 22 | 2.2% |

---

## 6. Sprint Retrospectives

### 6.1 Retrospective Format

We used the **Start-Stop-Continue** format for all retrospectives.

### 6.2 Sprint 3 Retrospective (ML Development)

**Start:**
- More frequent commits during training experiments
- Document hyperparameters in logs
- Share model training progress in standups

**Stop:**
- Training models without clear evaluation criteria
- Working in isolation on experimental code
- Overcomplicating model architectures initially

**Continue:**
- Using Jupyter notebooks for exploration
- Regular sync on data format standards
- Saving intermediate model checkpoints

**Action Items:**
- [x] Create model training log template
- [x] Add TensorBoard integration
- [x] Establish baseline metrics before experimenting

### 6.3 Sprint 5 Retrospective (API & Auth)

**Start:**
- API documentation (OpenAPI/Swagger)
- Integration tests for auth flows
- More pair programming sessions

**Stop:**
- Implementing without clear API contracts
- Last-minute debugging sessions
- Scope creep on "nice-to-have" features

**Continue:**
- Frontend/Backend sync meetings
- Using TypeScript types from backend schemas
- Code review before merge

**Action Items:**
- [x] Generate OpenAPI docs automatically
- [x] Create shared types package
- [x] Add CI check for type consistency

### 6.4 Sprint 7 Retrospective (User Testing)

**Start:**
- Recording user testing sessions (with consent)
- Prioritizing issues by frequency
- A/B testing for UI changes

**Stop:**
- Making assumptions about user behavior
- Ignoring "minor" usability issues
- Delaying bug fixes

**Continue:**
- Daily standups for quick sync
- Quick iteration cycles on feedback
- Testing on multiple devices

**Action Items:**
- [x] Create user feedback tracking board
- [x] Implement analytics for feature usage
- [x] Schedule follow-up testing session

---

## 7. Lessons Learned

### 7.1 Technical Lessons

| Lesson | Impact | Future Recommendation |
|--------|--------|----------------------|
| **Seq2Point outperformed Seq2Seq for NILM** | High | Start with simpler architectures |
| **Weighted loss functions critical for class imbalance** | High | Research domain-specific losses early |
| **API contracts should be defined early** | Medium | Use OpenAPI spec from Sprint 1 |
| **Docker Compose simplifies local development** | High | Containerize from the start |
| **TypeScript catches errors before runtime** | Medium | Enforce strict type checking |

### 7.2 Process Lessons

| Lesson | Impact | Future Recommendation |
|--------|--------|----------------------|
| **2-week sprints worked well for our team size** | High | Keep sprint duration |
| **Mid-sprint scope changes disrupted velocity** | Medium | Stricter sprint commitment |
| **User testing revealed unexpected issues** | High | Test earlier and more often |
| **Documentation during development saves time** | Medium | Document as you build |
| **Pair programming accelerated complex tasks** | Medium | Schedule regular sessions |

### 7.3 Communication Lessons

| Lesson | Impact | Future Recommendation |
|--------|--------|----------------------|
| **Daily standups kept everyone aligned** | High | Continue synchronous standups |
| **Async communication worked for simple updates** | Medium | Use Slack for non-urgent items |
| **Demo sessions built stakeholder confidence** | High | Regular demos to stakeholders |
| **Clear Definition of Done prevented disputes** | Medium | Refine DoD each sprint |

### 7.4 What We Would Do Differently

1. **Start user testing in Sprint 3** instead of Sprint 7
2. **Invest more in automated testing** from the beginning
3. **Create data mocking strategy** earlier for frontend development
4. **Establish code style guidelines** in Sprint 1
5. **Plan for deployment earlier** to catch environment issues

### 7.5 Team Achievements

- ✅ Delivered functional NILM web application
- ✅ Achieved 94% user testing task completion rate
- ✅ Maintained consistent velocity throughout project
- ✅ Successfully integrated ML models with production backend
- ✅ Deployed to cloud infrastructure (Cloudflare + Railway)
- ✅ Completed comprehensive documentation

### 7.6 Risks Encountered and Mitigations

| Risk | Likelihood | Impact | Mitigation | Outcome |
|------|------------|--------|------------|---------|
| ML model accuracy insufficient | Medium | High | Multiple architecture testing | Achieved acceptable accuracy |
| Data pipeline performance issues | Medium | Medium | Implemented caching layer | No production issues |
| Team member unavailability | Low | High | Cross-training on components | Minimal impact |
| Scope creep | High | Medium | Strict sprint boundaries | Managed effectively |

---

## Summary

The NILM Energy Monitor project successfully completed 8 sprints over 16 weeks, delivering a functional web application with AI-powered energy disaggregation. The team completed 183 of 220 planned story points (83%), with remaining items moved to the future backlog as non-critical enhancements.

Key success factors:
- Clear sprint goals aligned with academic milestones
- Regular retrospectives driving continuous improvement
- Strong technical foundation established early
- User testing informing meaningful iterations

The project demonstrates effective application of Agile/Scrum methodology in an academic context with a focus on working software and iterative improvement.

---

*Document Version: Final | Last Updated: January 2026*
