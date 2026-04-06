# Port Instructions for Local Development Environment

This document defines the standardized port allocation scheme for all local projects.

---

## Port Range Structure

Ports are grouped by service type:

| Port Range | Service Type              |
| ---------- | ------------------------- |
| 8000       | Global overview dashboard |
| 80xx       | Project web dashboards    |
| 81xx       | FastAPI services          |
| 82xx       | Databases                 |
| 89xx       | LLM                       |


Each project is assigned a unique two-digit identifier that maps consistently across services.

Pattern:

```
Project ID: NN
Web Dashboard: 80NN
FastAPI Service: 81NN
Database: 82NN
LLM : 89NN
```

Example:

Project ID 23 → "project x"

```
8023 → Web dashboard (project x)
8123 → FastAPI service (project x)
8223 → Database (project x)
8923 → LLM (project x)
```

---

## Example Multi‑Project Allocation

### Project 23 — project x

```
8023 → Web dashboard
8123 → FastAPI service
8223 → Database
```

### Project 24 — project y

```
8024 → Web dashboard
8124 → FastAPI service
8224 → Database
```

### Project 25 — project z

```
8025 → Web dashboard
8125 → FastAPI service
8225 → Database
```

---

## Global Services

Reserved ports:

```
8000 → Overview dashboard across all applications
```

This dashboard aggregates:

* project status
* running services
* links to dashboards
* links to FastAPI docs

---

## Naming Convention

Projects follow this format:

```
NN project-name
```

Example:

```
23 project-x
24 project-y
25 project-z
```

This ensures:

* predictable routing
* consistent service discovery
* easy mental mapping
* scalable local infrastructure

---

## Recommended Folder Structure

Example:

```
23-project-x/
 ├── dashboard/
 ├── api/
 ├── database/
 └── config/
```

Optional service mapping file:

```
ports.env
```

Example:

```
PROJECT_ID=23
DASHBOARD_PORT=8023
FASTAPI_PORT=8123
DATABASE_PORT=8223
```

---

## Benefits of This System

* predictable ports
* no collisions
* easy scaling to 99 projects
* simple automation support
* clean reverse-proxy compatibility
* works well with Docker and local-only setups

---

## Future Extensions (Optional)

If additional services are needed later:

Suggested reserved ranges:

```
83xx → background workers
84xx → vector databases
85xx → experimental services
86xx → admin tools
```

---

## Project 30 Allocation

For this project, the following ports are allocated following the 8x30<environment> pattern:

| Port | Service Type              | Docker Service   |
| ---- | ---------------------- | ------------- |
| 8030 | Web dashboard          | brain-frontend |
| 8130 | FastAPI service       | brain-api      |
| 8230 | Database              | (reserved)    |
| 8330 | Background workers    | (reserved)    |
| 8430 | Vector database       | (reserved)    |
| 8530 | Experimental services | (reserved)    |
| 8630 | Admin tools           | (reserved)    |

Pattern: `8` + service_category + project_number

- Service category 0 = web dashboard (80xx)
- Service category 1 = FastAPI (81xx)
- Service category 2 = database (82xx)
- Service category 3 = background workers (83xx)
- Service category 4 = vector databases (84xx)
- Service category 5 = experimental (85xx)
- Service category 6 = admin tools (86xx)
