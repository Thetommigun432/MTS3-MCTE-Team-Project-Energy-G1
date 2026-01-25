NILM Energy Monitor – Project Management Documentation

Document Version: Final
Project: NILM Energy Monitor
Date: January 2026
Methodology: Agile Scrum
Course: Team Project (MTS3)
Client: Jeroen De Baets (Energy Lab / Howest)

1. Methodology Overview
1.1 Agile Scrum Framework

The NILM Energy Monitor project was developed using the Agile Scrum methodology, adapted to the intensive structure of the “Team Project” academic module. Given the limited execution time and the strong focus on delivering a working prototype, Scrum was selected for its ability to support rapid iteration, frequent feedback, and incremental delivery of value.

The overall development process was divided into distinct phases. An initial planning and research phase took place in November 2025, during which the problem domain was analyzed, architectural decisions were made, and the initial product backlog was created. This phase was followed by three intensive development weeks in January 2026, during which the full system was implemented, integrated, tested, and deployed.

Due to the short execution window, the team adopted weekly sprints instead of the more common two-week cadence. This decision enabled fast feedback loops and clear short-term goals. Daily stand-up meetings were held synchronously every morning at 09:00 to align on daily objectives, identify blockers, and coordinate work across frontend, backend, and machine learning components. Sprint reviews were organized at key milestones, notably at the end of the second week for the interim demo and at the end of the third week for the final delivery. Retrospectives were conducted weekly to reflect on both technical challenges and team collaboration, allowing continuous improvement throughout the project.

1.2 Team Structure

The project team consisted of three members who worked collaboratively while maintaining clear areas of expertise. Tommaso Pioda focused primarily on system architecture and integration, taking responsibility for backend development using FastAPI, database management with InfluxDB and Supabase, containerization through Docker, and cloud deployment. Rodrigo Sousa was responsible for frontend development and user experience, implementing the React-based interface with Vite, designing dashboards and visualizations, and conducting user testing sessions. Mirko Keller specialized in artificial intelligence and data science, leading the development and training of the NILM models using PyTorch, handling data preprocessing, and designing the inference pipeline.

The client and Product Owner role was fulfilled by Jeroen De Baets from Energy Lab at Howest, who provided domain expertise, requirements clarification, and feedback throughout the project. Academic guidance and evaluation were supported by the coaches Marie Dewitte, Wouter Gevaert, and Frederik Waeyaert.

2. Project Timeline

The project followed a strict academic timeline defined by the Team Project module. During the introduction week in early November 2025, the team focused on domain research, architectural design, and backlog creation. This was followed by a holiday break period, during which team members conducted independent study, explored relevant technologies, and prepared their local development environments.

Active development began in January 2026 with three consecutive weekly sprints. The first sprint focused on building the technical foundations, including the data pipeline, initial machine learning models, and a basic user interface. The second sprint was centered around full system integration and culminated in the interim demo, which accounted for 20% of the final evaluation. The third sprint emphasized user testing, refinement, documentation, and deployment, preparing the project for the final jury presentation held at the end of January, which represented 50% of the final grade.

3. Sprint Planning and Backlogs
3.1 Introduction Week (Sprint 0)

The introduction week functioned as Sprint 0 and was dedicated to defining the problem space and overall solution architecture. During this phase, the team selected the Seq2Point approach as the core NILM model architecture and finalized the technology stack, including FastAPI for the backend, React for the frontend, InfluxDB for time-series data storage, and Docker for containerization. An initial backlog was created using Trello and later migrated to GitHub, providing a structured overview of features, tasks, and priorities.

3.2 Sprint 1: Foundation and MVP

The first development sprint, held from January 5 to January 9, focused on establishing a functional minimum viable product. Key objectives included implementing the data ingestion and preprocessing pipeline, training the first baseline NILM model, and setting up the basic frontend structure. During this sprint, the team successfully implemented data loaders for UK-DALE and REFIT datasets, configured InfluxDB with read and write API wrappers, scaffolded the React frontend with Tailwind CSS and Recharts, trained a baseline CNN-based Seq2Point model, and created a Docker Compose setup to orchestrate the full stack. By the end of the sprint, the project had a working end-to-end foundation.

3.3 Sprint 2: Integration and Interim Demo

The second sprint, running from January 12 to January 16, focused on integrating the machine learning models with the backend and frontend to produce a coherent, demonstrable system. A key milestone of this sprint was the interim presentation. During this phase, the inference endpoint was implemented, connecting FastAPI to the PyTorch models, and real-time energy visualizations were added to the frontend. Appliance-level breakdown charts were introduced, and inference speed was optimized to support near real-time predictions. The sprint concluded with the preparation and delivery of the interim presentation, during which the team successfully demonstrated real-time energy disaggregation from a synthesized aggregate signal.

3.4 Sprint 3: Polish and Validation

The final sprint, from January 19 to January 23, was dedicated to validation, refinement, and delivery readiness. User testing sessions were conducted with five participants to evaluate usability and clarity. Based on feedback, confidence score visualizations were added to improve transparency of model predictions. Final machine learning models were trained for five appliances, and the application was deployed using Cloudflare and Railway. Documentation, including the functional analysis and user manual, was finalized to support the final jury evaluation.

4. Burndown Analysis

The burndown analysis tracked remaining effort in hours across the three development weeks. At the start of January, the project had an estimated workload of approximately 120 hours. By the end of the first week, progress was largely on track, although initial environment setup required more time than expected. During the second week, a temporary plateau was observed due to integration challenges between PyTorch-based inference and FastAPI’s asynchronous execution model. This issue was resolved through focused debugging and pair programming. By the interim demo, approximately 80% of the MVP functionality had been completed, and scope was consciously adjusted to focus on five appliances instead of ten. At the end of the third week, all critical tasks were completed, resulting in zero remaining high-priority items. The team maintained a combined velocity of approximately 80 to 90 productive hours per week.

5. Time Tracking

Across the introduction phase and three execution weeks, the team invested approximately 280 total hours. Roughly half of this time was dedicated to software development, including frontend, backend, and infrastructure work. Machine learning training and experimentation accounted for just over one-fifth of the total effort, reflecting the complexity of NILM model development. Documentation and reporting represented around 14% of the workload, while meetings, reviews, and daily stand-ups accounted for 7%. User testing and UX improvements also represented approximately 7% of the total effort. On average, each student contributed between 90 and 95 hours over the course of the project.

6. Sprint Retrospectives

Sprint retrospectives were conducted weekly to assess both technical progress and team dynamics. After Sprint 1, the team identified the importance of shared type definitions for API responses to avoid frontend-backend mismatches and the need to explicitly document machine learning model input shapes. One key improvement was stopping the practice of committing large model files to version control, instead moving them to external storage and updating the .gitignore.

Following the interim presentation in Sprint 2, feedback from the jury emphasized the need to clearly communicate the confidence of model predictions and to highlight the business value of appliance-level energy insights. As a result, the team added a confidence interval visualization component and introduced a cost estimation view in the dashboard during Sprint 3.

7. Conclusion

The NILM Energy Monitor project successfully met the objectives of the Team Project module within an aggressive three-week development timeline. By applying a structured Scrum framework with weekly sprints and clearly defined milestones, the team was able to manage the complexity of integrating deep learning models with a modern web application stack. The final outcome is a deployed, user-tested NILM application that fulfills the client’s goal of enabling smarter and more cost-effective analysis of energy consumption.