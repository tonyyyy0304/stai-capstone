Introduction to Agentic AI (STAI100)

**Midterm Capstone**

Project Specification

Due: Week 9  ·  Teams of 3–4 students

# **1\. Overview**

The Midterm Capstone is the first major integration milestone of the course. Over Weeks 1–7, you have built individual components of an agentic AI system; prompt engineering, structured outputs, retrieval-augmented generation (RAG), memory, guardrails, tool use, and agent loops. The Midterm Capstone asks you to bring these components together into a single, coherent, working application grounded in a real business problem.

Your team will design, build, evaluate, and present an end-to-end agentic system. The deliverable is not a prototype or a demo script; it is a deployable application with documented architecture, observable behavior, and evidence of systematic testing.

| What you will build A working agentic AI application solving a real business problem Integrating modules from Weeks 1–7 (at least 2 modules per team member) Accessible via a web UI and an API endpoint Deployed with basic LLMOps monitoring Documented with a technical write-up (≥2,000 words) and a clean code repository |
| :---- |

# **2\. Learning Objectives**

By completing this capstone, students will be able to:

* Integrate multiple agentic components (RAG, memory, guardrails, tool use) into a single coherent system

* Justify architectural decisions and communicate trade-offs clearly

* Evaluate system reliability through structured experiments and document findings

* Package and deploy an AI application with observability tooling

* Present technical work to a non-specialist audience using a business framing

# **3\. Project Requirements**

## **3.1  Technical Requirements**

* Working, end-to-end agentic AI application

* Demonstrates modules from Weeks 1–7; minimum 2 modules per team member (e.g., a 3-person team covers at least 6 modules)

* Accessible via a web UI (e.g., Streamlit, Gradio)

* Exposes an API endpoint (REST)

* Deployed with basic LLMOps monitoring (e.g. MLFlow) 

* Containerized with a Dockerfile and documented build/run instructions

## **3.2  Team Requirements**

* Teams of 3–4 students; self-formed by Week 6

* Each member must own and be able to explain at least 2 modules

* All members participate in the live presentation

## **3.3  Deliverables**

Submit all of the following by the Week 9 deadline:

| Deliverable | Details |
| :---- | :---- |
| **Live Presentation** | 10–15 minutes including Q\&A; slides required |
| **Technical Write-up** | ≥2,000 words covering business case, methodology, architecture, experiments, and retrospective |
| **Source Code Repository** | GitHub (or equivalent .zip) repo with README, Dockerfile, and inline documentation |
| **Working Demo** | Live, accessible demo during presentation (no pre-recorded video substitutes) |

# **4\. Module Checklist**

Each team member is responsible for at least two modules from the list below. During the presentation, every member should be prepared to walk through the modules they owned; explaining design decisions, showing relevant code, and discussing evaluation results.

| Module | Description |
| :---- | :---- |
| **Prompt Engineering** | Design and iterate on system prompts; apply few-shot, chain-of-thought, and structured prompt patterns |
| **Structured Outputs** | Return typed, schema-validated responses (JSON, Pydantic, etc.) for downstream consumption |
| **Disambiguation** | Detect ambiguous inputs and clarify intent before proceeding with tool calls or generation |
| **RAG** | Retrieve relevant context from a vector, SQL, or graph store and ground model responses in retrieved data |
| **Memory** | Maintain short-term session memory and/or long-term persistent memory across conversations |
| **Guardrails** | Implement input/output validation, topic filtering, and safety checks |
| **ReAct Agent** | Implement a reasoning \+ acting loop where the model plans and executes steps iteratively |
| **SQL Agent** | Generate and execute SQL queries against a relational database based on natural language |
| **Tool Use** | Integrate at least one external tool or API (search, weather, calendar, etc.) |
| **Chat UI** | Build a functional conversational interface (e.g., Streamlit, Gradio) |
| **API Endpoint** | Expose the agent via a REST API endpoint |
| **LLMOps Monitoring** | Log traces, latency, token usage, and errors using an observability tool (e.g., MLFlow) |
| **Dockerization** | Package the application in a Dockerfile with documented build and run instructions |

Note: Prompt Engineering is foundational and expected to appear throughout the system. It counts as a module only when it is explicitly designed, iterated on, and evaluated (e.g., ablation across prompt variants).

# **5\. Presentation Structure**

Presentations are 15-20 minutes followed by Q\&A.   
(5 mins per person, i.e. 15 mins for 3-person team)

Structure your slides around the following sections:

| \# | Section | What to Cover |
| :---- | :---- | :---- |
| 1 | **Business Use Case** | The problem, the target user, and why an agentic approach is appropriate |
| 2 | **Architecture & Methodology** | System diagram, component breakdown, data model and flow, and technology stack |
| 3 | **Live Demo** | Walkthrough of the working application; demonstrate key agentic behaviors |
| 4 | **Experiment Findings** | What you tested, how you measured it, and what the results showed |
| 5 | **Retrospective** | What worked, what did not, what you would do differently, and open issues |

# **6\. Grading Rubric**

| Criterion | Weight | Description |
| :---- | :---- | :---- |
| **Technical Depth and Correctness** | 30% | Correct use of RAG, memory, guardrails, tool use, and agent patterns; architecture matches implementation |
| **System Architecture and Design Quality** | 25% | Clear separation of concerns, appropriate component selection, coherent data flow |
| **Eval Results and Reliability Demonstration** | 20% | Evidence-based testing, edge case handling, documented failure modes and mitigations |
| **Presentation Quality and Live Demo** | 15% | Clear communication of business case, smooth walkthrough, answers to Q\&A |
| **Code Quality, Documentation, and README** | 10% | Readable code, inline comments, complete README with setup instructions and architecture overview |

Note: A functional live demo is expected. A demo that fails to run during presentation will affect the “Presentation Quality and Live Demo” criterion. Prepare a fallback (e.g., screen recording) and disclose it upfront.

# **7\. Course Grading Context**

The Midterm Capstone contributes 30% of your final course grade. The table below shows how all assessments are weighted.

| Assessment | Weight | Description |
| :---- | :---- | :---- |
| **Weekly Homework** | 25% | Lab exercises submitted as Jupyter notebooks with documentation |
| **Midterm Capstone (Week 9\)** | 30% | Working agentic system demonstrating components from Weeks 1 to 7 |
| **Final Capstone Project (Week 14\)** | 40% | End-to-end agentic solution with CV/DS model integration |
| **Participation & Peer Review** | 5% | In-class engagement, capstone dry-run feedback, Week 14 peer evaluations |

# **8\. Choosing a Good Problem**

Not every problem benefits from an agentic approach. Use the criteria below to evaluate your proposed use case before committing.

| ✅  Good Fits for Agentic AI Multi-step reasoning over external tools or APIs Processes unstructured data (PDFs, audio, images, web pages) Needs memory or context across a conversation Real users with measurable success criteria Workflow currently done manually and repeatedly | ❌  Poor Fits for Agentic AI Single-call Q\&A ;  a well-crafted prompt would suffice Pure CRUD apps using an LLM as a thin wrapper Tasks where deterministic code already wins Problems with no ground truth or evaluation framework Safety-critical workflows without a human-in-the-loop |
| :---- | :---- |

# **9\. Suggested Use Cases**

The table below lists suggested use cases organized by retrieval type (API, SQL, Vector DB, Graph DB). These are starting points; you are encouraged to propose your own problem, as long as it meets the criteria in Section 8\. Teams building toward the Final Capstone may also want to consider use cases that can later incorporate a CV or DS model.

Refer to the sheet provided in-class for the updated list, and to note down the use case that your group will be taking.

| RAG Type | Use Case | Example Query |
| :---- | :---- | :---- |
| **API** | Weather / Climate Data | "Give me the months where rainfall exceeded 150 mm in 2023." |
|  | Solar Irradiation (NREL/NSRDB) | "What is the monthly average irradiation from Feb to June 2022?" |
|  | Stock Price Monitoring | "What were the min/max prices of $GLO (Globe Telecom) from May to June?" |
|  | Valuation Modeling | "What is the current DCF-based valuation of $COMPANY?" |
|  | SEC / Tax Filing Research | "Retrieve all relevant tax filings and registration documents for \<Company\>." |
|  | HR Scheduling | "From the top 20 candidates, send email invites to the 5 we haven't interviewed yet." |
| **SQL** | Virtual Finance Analyst | "List the top 5 departments with highest operational expenses last quarter." |
|  | Company L\&D Training Tracker | "What mandatory trainings do I need to complete for the Cloud & Data capability?" |
|  | Supply Chain / Inventory | "Show the top 10 SKUs with the highest lead times this month." |
|  | Customer Segmentation | "How many segments should I create ad campaigns for based on 2025 data?" |
|  | Hospital Directory | "Which cardiologists are available on Tuesdays and accredited with MediCard?" |
| **Vector DB** | Healthcare Procedures Q\&A | "I have a lipid profile test at 10 AM ;  what time should I last eat?" |
|  | Medical Pricing | "How much is a CT scan, and what portion is covered by PhilHealth?" |
|  | Junior QA Assistant | "Generate a test plan for a virtual assistant built for a car dealership." |
|  | Digital Trends Research | "Which social media platform should we focus on for maximum reach in PH 2026?" |
|  | Website Q\&A (External) | "What Stratpoint projects were related to retail?" |
|  | HR Onboarding | "What do I need to do on my first day?" |
|  | Legal / IP Assistant | "What are the penalties for intellectual property fraud?" |
|  | Style Guide Checker | "Review this PR against the Google Style Guide." |
| **Graph DB** | Address Resolution | "Which cities fall under the NCR region?" |
|  | Disease Propagation | "If Quezon City is locked down, which neighboring cities are at risk?" |
|  | HR Org Chart / Mentorship | "Who is my tech mentor, and who is my career mentor?" |
|  | Recommendation Engine | "Which movies are similar to Movie X based on user ratings?" |

# **10\. Path to the Final Capstone (Optional)**

The Final Capstone (Week 14, 40% of course grade) extends your Midterm Capstone with a Computer Vision or Data Science model integration. Building on your Midterm project is encouraged but not required; teams may start fresh for the Final.

If you plan to build on this project, consider selecting a use case and architecture that leaves room for model integration. For example:

* A supply chain agent (Midterm) that incorporates a demand forecasting model (Final)

* A hospital triage assistant (Midterm) that adds medical image analysis (Final)

* A customer segmentation tool (Midterm) that plugs in a classification or clustering model (Final)

| 📷  Computer Vision Track Object detection and OCR pipelines Multimodal agents (image \+ text) Document understanding (forms, receipts) Example: LLM \+ CV for license plate retrieval, object detection timestamps | 📊  Data Science Track RAG-driven NLP pipelines Analytics and report automation Tool-using research agents Example: LLM \+ DS for forecasting, segmentation, or financial modeling |
| :---- | :---- |

Discuss your Final Capstone direction with your instructor during or after the Midterm presentation if you want early feedback.

# **11\. Submission Checklist**

Before submitting, confirm that your team has completed all of the following:

| ✓ | Item |
| :---- | :---- |
| □ | Working demo accessible via web UI and API endpoint |
| □ | At least 2 modules per team member integrated and demonstrable |
| □ | LLMOps monitoring configured (traces, latency, token usage visible) |
| □ | Dockerfile builds and runs cleanly with a single command |
| □ | README includes: project overview, setup instructions, architecture diagram, and module ownership table |
| □ | Technical write-up (≥2,000 words) submitted as PDF or markdown |
| □ | Presentation slides finalized and submitted |
| □ | All team members prepared to answer questions on their owned modules |

