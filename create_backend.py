from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from mcp import build_mcp_agent, build_mcp_tools

PLANNING_SYSTEM_PROMPT = """\
You are a software architect and planning assistant. Your task is to analyze a development request and create a detailed implementation plan.

Your response MUST be a single, valid JSON object enclosed in a ```json code block. Do not include any text or explanations before or after the JSON block.

The JSON object must have a single root key: "implementation_plan".

The "implementation_plan" object must contain the following keys:
1. "files": An ordered list of file objects. Each object must have "path" (string) and "description" (string). The order of files should represent the ideal implementation sequence.
2. "dependencies": A list of strings representing external dependencies to be installed (e.g., via pip or npm).
3. "notes": A string for any additional implementation notes or high-level guidance for the developer.

**CRITICAL RULES:**
- Your entire response must be ONLY the JSON object inside a ```json code block.
- Do not add any comments inside the JSON.
- Ensure all JSON strings are correctly formatted and escaped.
- The plan should be thorough and cover all necessary files for a complete, production-ready implementation.

Example of a valid response:
```json
{
    "implementation_plan": {
        "files": [
            {"path": "requirements.txt", "description": "Python dependencies"},
            {"path": "app/database.py", "description": "Database setup and connection logic"},
            {"path": "app/main.py", "description": "FastAPI application entry point"}
        ],
        "dependencies": ["fastapi", "uvicorn", "pydantic"],
        "notes": "The project will use FastAPI and SQLite. Start with setting up dependencies and the database."
    }
}
```
"""

IMPLEMENTATION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert software developer implementing a batch of files as part of a larger project.

PROJECT CONTEXT:
{project_description}

IMPLEMENTATION PLAN:
{implementation_plan}

CURRENT FOLDER STRUCTURE:
{folder_structure}

CURRENT BATCH OF TASKS:
You are to implement all of the following files in this batch:
{file_batch_details}

INSTRUCTIONS:
1. Read the content of any relevant files to understand the context.
2. Implement ALL the files specified in CURRENT BATCH OF TASKS. Create each file with its complete, production-ready code.
3. Include all necessary imports and dependencies for each file.
4. Follow best practices and proper error handling.
5. Make sure the code integrates well with the overall project structure.
6. Do not implement other files—focus only on the files in the current batch.

Use the available tools to create the files.

AND MOST IMPORTANTLY: DO NOT IMPLEMENT TESTS IN THIS PHASE; THERE IS A SEPARATE PHASE FOR TESTING. Just do the implementation
"""

TESTING_SYSTEM_PROMPT = """\
You are a software quality assurance engineer. Your task is to test the implemented code by creating and running a comprehensive test suite.

INSTRUCTIONS:
1. Create a `test_backend.py` file to test all API endpoints. Dont write a single test function write multiple ones to test the endpoint functionaility independent
2. The tests must be thorough and cover all functionalities. 
3. Run the tests and analyze the results.
4. If there are any failures, debug the code, fix the issues, and run the tests again.
5. Repeat the process until all tests pass.
6. Once all tests pass, provide a summary of the results.

Use the available tools to create the test file, run the tests, and debug the code.
"""

def build_planning_agent(model: BaseChatModel) -> Runnable:
    """Build an agent specifically for the planning phase"""
    return build_mcp_agent(model, [], PLANNING_SYSTEM_PROMPT)

def build_implementation_agent_for_batch(model: BaseChatModel, tools: List, project_description: str, implementation_plan: str, file_batch: List[Dict[str, str]], folder_structure: str) -> Runnable:
    """Build an agent specifically for implementing a batch of files"""
    file_batch_details = ""
    for file_info in file_batch:
        file_batch_details += f'- path: {file_info["path"]}\\n  description: {file_info["description"]}\\n'

    system_prompt = IMPLEMENTATION_SYSTEM_PROMPT_TEMPLATE.format(
        project_description=project_description,
        implementation_plan=implementation_plan,
        file_batch_details=file_batch_details,
        folder_structure=folder_structure
    )
    return build_mcp_agent(model, tools, system_prompt)

def build_testing_agent(model: BaseChatModel, tools: List) -> Runnable:
    """Build an agent specifically for the testing phase"""
    return build_mcp_agent(model, tools, TESTING_SYSTEM_PROMPT)

def mcp_fast_iterative(model: BaseChatModel, test_llm: BaseChatModel, planning_llm: BaseChatModel, workspace_root: Union[str, Path], project_description: str, batch_size: int = 1) -> Dict[str, Any]:
    """
    Fast implementation using a three-phase approach with batched implementation.
    1. Planning phase: Create implementation plan.
    2. Implementation phase: Iterate through files in batches of N.
    3. Testing phase: Create and run tests until they all pass.
    """
    print("=== PHASE 1: PLANNING ===")
    
    planning_agent = build_planning_agent(planning_llm)
    planning_result = planning_agent.invoke({
        "messages": [{"role": "user", "content": f"Create an implementation plan for this project:\n\n{project_description}"}]
    })
    
    plan_content = planning_result["messages"][-1].content
    print(f"Planning result:\n{plan_content}")
    
    try:
        import re
        json_match = re.search(r"```json\\s*({.*?})\\s*```", plan_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'{.*}', plan_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                raise ValueError("No JSON object found in the planning response.")

        plan_json = json.loads(json_str)
        implementation_plan = plan_json.get("implementation_plan", plan_json)
        files_to_implement = implementation_plan["files"]
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse implementation plan: {e}\nContent: {plan_content}")
    
    plan_path = Path(workspace_root) / "implementation_plan.json"
    with open(plan_path, "w") as f:
        json.dump(implementation_plan, f, indent=2)
    print(f"Implementation plan saved to {plan_path}")

    print(f"Implementation plan: {implementation_plan}")
    
    print("\n=== PHASE 2: IMPLEMENTATION ===")
    
    tools = build_mcp_tools(workspace_root)
    config = {"configurable": {"workspace_root": str(Path(workspace_root).resolve())}}
    
    implementation_results = []
    
    file_batches = [files_to_implement[i:i + batch_size] for i in range(0, len(files_to_implement), batch_size)]

    for batch_index, batch in enumerate(file_batches, 1):
        print(f"\n--- Implementing batch {batch_index}/{len(file_batches)} ---")
        
        fs_list_tool = next(t for t in tools if t.name == "fs_list")
        folder_structure = fs_list_tool.invoke({"path": "."}, config=config)
        
        implementation_agent = build_implementation_agent_for_batch(
            model=model,
            tools=tools,
            project_description=project_description,
            implementation_plan=json.dumps(implementation_plan, indent=2),
            file_batch=batch,
            folder_structure=folder_structure
        )
        
        configured_agent = implementation_agent.with_config(config)
        
        batch_details_for_prompt = "\n".join([f'- `{f["path"]}`: {f["description"]}' for f in batch])
        file_prompt = f"""\
Please implement all files in the current batch.

Batch files:
{batch_details_for_prompt}

Read any relevant existing files to understand context, then implement each file with complete, production-ready code."""
        
        try:
            result = configured_agent.invoke({
                "messages": [{"role": "user", "content": file_prompt}]
            }, config={"recursion_limit": 250})
            
            final_message = result["messages"][-1]
            
            implementation_results.append({
                "batch": batch_index,
                "files": [f["path"] for f in batch],
                "status": "success",
                "result": final_message.content
            })
            
        except Exception as e:
            print(f"Error implementing batch {batch_index}: {e}")
            implementation_results.append({
                "batch": batch_index,
                "files": [f["path"] for f in batch],
                "status": "error",
                "error": str(e)
            })

    print("\n=== PHASE 3: TESTING ===")
    
    testing_agent = build_testing_agent(test_llm, tools)
    configured_testing_agent = testing_agent.with_config(config)
    
    try:
        test_prompt = "Create a `test_backend.py` file, then run the tests. If they fail, debug the code and repeat until all tests pass."
        testing_result = configured_testing_agent.invoke({
            "messages": [{"role": "user", "content": test_prompt}]
        }, config={"recursion_limit": 250})
        
        final_testing_message = testing_result["messages"][-1]
        
        testing_phase_result = {
            "status": "success",
            "final_message": final_testing_message.content
        }
        
    except Exception as e:
        print(f"Error during testing phase: {e}")
        testing_phase_result = {
            "status": "error",
            "error": str(e)
        }
        
    return {
        "planning_phase": {
            "plan": implementation_plan,
            "raw_response": plan_content
        },
        "implementation_phase": {
            "results": implementation_results,
            "files_implemented": sum(len(r['files']) for r in implementation_results if r["status"] == "success"),
            "files_failed": sum(len(r['files']) for r in implementation_results if r["status"] == "error")
        },
        "testing_phase": testing_phase_result
    }

if __name__ == "__main__":
    message = """
                      Implement this backend. It must be runnable without errors. If you use pydantic remember to use pydantic-settings and pydantic bigger than 2.0.0 as this will be installed. dont use old syntax. Make it production ready dont simplify. :
                     Assumptions

Single organization per deployment by default, but the schema includes org_id to support multiple teams in the future. All queries are automatically scoped by org_id.

Email sending is abstracted behind a provider-agnostic service (e.g., SMTP/API). We persist an outbox (email_queue) and delivery events; an external worker can be attached later, but the API exposes enqueue + status now.

File storage is local disk (./data/storage) with metadata in DB; swap-able via env variable. Virus scanning and OCR are out of scope.

Calendar “integration” means internal conflict detection + optional external provider reference fields; actual provider OAuth/webhooks are out of scope.

Bulk import processes synchronously within request limits (chunked) for MVP; large-file background processing is planned (table support present).

RBAC roles: admin, recruiter, hiring_manager, interviewer, viewer. Fine-grained permission checks are hooks inside services.

Currencies default to the organization base currency (env ORG_CURRENCY, default USD).

Costs are stored as integer minor units (e.g., cents).

Offer approval workflow is a simple sequential approval chain (ordered approvers).

“Custom report builder” returns server-side aggregated data with validated filters, grouping, and CSV export.

Backend API Plan

Conventions:

Base: /api/v1

Pagination: ?page=&page_size=

Envelope: { items, page, page_size, total }

Errors: { "error": { "code", "message", "details?" } }

Auth: JWT (access+refresh) in HTTP-only cookies; CSRF via X-CSRF-Token for unsafe methods.

Idempotency: pass Idempotency-Key header on POST to create-imports, send-email, create-offer.

Auth & Session

POST /api/v1/auth/register — register first admin (if no users). Public once, then admin-only.
Schemas: AuthRegisterRequest → UserResponse

POST /api/v1/auth/login — email+password login. Public.
AuthLoginRequest → AuthLoginResponse

POST /api/v1/auth/refresh — rotate refresh token. Cookie auth (refresh).
null → AuthRefreshResponse

POST /api/v1/auth/logout — revoke tokens. Authenticated.
null → SuccessResponse

GET /api/v1/auth/csrf — issue CSRF token (double-submit cookie). Public.
null → CsrfTokenResponse

GET /api/v1/auth/me — current user profile. Authenticated.
null → UserResponse

POST /api/v1/auth/change-password — change password. Authenticated.
ChangePasswordRequest → SuccessResponse

Users & RBAC

GET /api/v1/users — list users. Admin.
Query: role? → PaginatedUserResponse

POST /api/v1/users — create user (invite flow, temporary password). Admin.
UserCreateRequest → UserResponse

GET /api/v1/users/{user_id} — get user. Admin/self. → UserResponse

PUT /api/v1/users/{user_id} — update user profile/role. Admin.
UserUpdateRequest → UserResponse

DELETE /api/v1/users/{user_id} — deactivate. Admin. → SuccessResponse

Candidates

GET /api/v1/candidates — list/filter. Recruiter+.
Query: q, source, created_from,to, rating_min,max, tags, page, page_size, sort → PaginatedCandidateResponse

POST /api/v1/candidates — create. Recruiter+.
CandidateCreateRequest → CandidateResponse

GET /api/v1/candidates/{candidate_id} — fetch details (notes, docs counts inline). Recruiter+. → CandidateDetailResponse

PUT /api/v1/candidates/{candidate_id} — update. Recruiter+.
CandidateUpdateRequest → CandidateResponse

DELETE /api/v1/candidates/{candidate_id} — soft-delete. Admin/Recruiter. → SuccessResponse

POST /api/v1/candidates/{candidate_id}/score — manual score/rating. Recruiter/HiringMgr.
CandidateScoreRequest → CandidateScoreResponse

POST /api/v1/candidates/bulk/import — upload CSV/XLSX and import. Recruiter+.
multipart/form-data → ImportJobResponse

GET /api/v1/candidates/imports/{job_id} — import job status. Recruiter+. → ImportJobDetailResponse

Candidate Notes, Documents & Communication

GET /api/v1/candidates/{candidate_id}/notes — list. Recruiter+. → PaginatedNoteResponse

POST /api/v1/candidates/{candidate_id}/notes — add. Recruiter+.
NoteCreateRequest → NoteResponse

GET /api/v1/candidates/{candidate_id}/documents — list. Recruiter+. → PaginatedDocumentResponse

POST /api/v1/candidates/{candidate_id}/documents — upload resume/cover/portfolio. Recruiter+.
multipart/form-data → DocumentResponse

GET /api/v1/candidates/{candidate_id}/communications — list emails/updates. Recruiter+. → PaginatedCommunicationResponse

Jobs & Pipelines

GET /api/v1/jobs — list/filter. Recruiter+. → PaginatedJobResponse

POST /api/v1/jobs — create job posting. Recruiter/HiringMgr.
JobCreateRequest → JobResponse

GET /api/v1/jobs/{job_id} — details w/ pipeline stages. Recruiter/HiringMgr. → JobDetailResponse

PUT /api/v1/jobs/{job_id} — update. Recruiter/HiringMgr. → JobResponse

DELETE /api/v1/jobs/{job_id} — archive. Admin/Recruiter. → SuccessResponse

GET /api/v1/jobs/{job_id}/stages — list stages. Recruiter+. → PipelineStageListResponse

POST /api/v1/jobs/{job_id}/stages — add stage. Recruiter+.
PipelineStageCreateRequest → PipelineStageResponse

PUT /api/v1/jobs/{job_id}/stages/{stage_id} — rename/reorder. Recruiter+. → PipelineStageResponse

DELETE /api/v1/jobs/{job_id}/stages/{stage_id} — remove (if empty). Recruiter+. → SuccessResponse

Applications (Candidate ↔ Job)

GET /api/v1/applications — list/filter. Recruiter+. → PaginatedApplicationResponse

POST /api/v1/applications — create application (existing/new candidate). Recruiter+.
ApplicationCreateRequest → ApplicationResponse

GET /api/v1/applications/{app_id} — details + history. Recruiter+. → ApplicationDetailResponse

POST /api/v1/applications/{app_id}/move — move stage (triggers rules & sequences). Recruiter/HiringMgr.
ApplicationMoveRequest → ApplicationResponse

POST /api/v1/applications/{app_id}/reject — reject with reason. Recruiter/HiringMgr.
ApplicationRejectRequest → ApplicationResponse

Interviews & Feedback

GET /api/v1/interviews — list/filter by job/candidate. Recruiter+. → PaginatedInterviewResponse

POST /api/v1/interviews — schedule (conflict detection). Recruiter/HiringMgr.
InterviewCreateRequest → InterviewResponse

PUT /api/v1/interviews/{interview_id} — reschedule/update. Recruiter/HiringMgr. → InterviewResponse

DELETE /api/v1/interviews/{interview_id} — cancel. Recruiter/HiringMgr. → SuccessResponse

GET /api/v1/interviews/{interview_id}/feedback-form — fetch structured form. Interviewer+. → FeedbackFormResponse

POST /api/v1/interviews/{interview_id}/feedback — submit feedback. Interviewer.
FeedbackSubmitRequest → FeedbackResponse

GET /api/v1/interviews/{interview_id}/feedback — list feedbacks. Recruiter/HiringMgr. → PaginatedFeedbackResponse

Offers & Approvals

GET /api/v1/offers — list/filter (status, job). Recruiter/HiringMgr. → PaginatedOfferResponse

POST /api/v1/offers — create offer. Recruiter/HiringMgr.
OfferCreateRequest → OfferResponse

GET /api/v1/offers/{offer_id} — get offer + approvals. Recruiter/HiringMgr/Approver. → OfferDetailResponse

POST /api/v1/offers/{offer_id}/submit — start approval chain. Recruiter/HiringMgr. → OfferResponse

POST /api/v1/offers/{offer_id}/approve — approver action. Approver.
OfferApproveRequest → OfferResponse

POST /api/v1/offers/{offer_id}/reject — approver reject. Approver. → OfferResponse

POST /api/v1/offers/{offer_id}/withdraw — withdraw offer. Recruiter/HiringMgr. → OfferResponse

Email Templates & Sequences

GET /api/v1/email/templates — list. Recruiter+. → PaginatedEmailTemplateResponse

POST /api/v1/email/templates — create. Recruiter+.
EmailTemplateCreateRequest → EmailTemplateResponse

PUT /api/v1/email/templates/{template_id} — update. Recruiter+. → EmailTemplateResponse

DELETE /api/v1/email/templates/{template_id} — delete. Recruiter+. → SuccessResponse

GET /api/v1/email/sequences — list. Recruiter+. → PaginatedEmailSequenceResponse

POST /api/v1/email/sequences — create sequence & steps. Recruiter+.
EmailSequenceCreateRequest → EmailSequenceResponse

POST /api/v1/email/send — send single templated email. Recruiter+.
EmailSendRequest → EmailQueueResponse

Workflow Rules

GET /api/v1/workflows/rules — list. Recruiter+. → PaginatedWorkflowRuleResponse

POST /api/v1/workflows/rules — create rule. Recruiter+.
WorkflowRuleCreateRequest → WorkflowRuleResponse

PUT /api/v1/workflows/rules/{rule_id} — update/enable/disable. Recruiter+. → WorkflowRuleResponse

DELETE /api/v1/workflows/rules/{rule_id} — delete. Recruiter+. → SuccessResponse

Analytics & Reporting

GET /api/v1/analytics/funnel — conversion per stage/time window. Recruiter+.
Query: job_id?, from?, to? → FunnelAnalyticsResponse

GET /api/v1/analytics/time-to-hire — medians/averages. Recruiter+. → TimeToHireResponse

GET /api/v1/analytics/source-effectiveness — hires per source & cost. Recruiter+. → SourceEffectivenessResponse

GET /api/v1/analytics/hiring-manager-performance — cycle times/feedback SLAs. Recruiter+. → HiringManagerPerformanceResponse

POST /api/v1/reports/run — custom report builder. Recruiter+.
ReportRunRequest → ReportResultResponse

GET /api/v1/reports/{report_id}/export — CSV export. Recruiter+. → text/csv

Settings & Misc

GET /api/v1/settings — org settings. Admin. → SettingsResponse

PUT /api/v1/settings — update org settings. Admin.
SettingsUpdateRequest → SettingsResponse

GET /api/v1/health — liveness. Public. → HealthResponse

GET /api/v1/openapi.json — OpenAPI. Public.

Backend File Structure
app/
  __init__.py
  main.py
  config.py
  database.py
  dependencies.py
  core/
    __init__.py
    security.py
    errors.py
    logging.py
    rate_limit.py
    csrf.py
  sql/
    schema.sql
    seed.sql
    triggers.sql
  schemas/
    __init__.py
    common.py
    auth.py
    users.py
    candidates.py
    notes.py
    documents.py
    jobs.py
    pipelines.py
    applications.py
    interviews.py
    feedback.py
    offers.py
    emails.py
    workflows.py
    analytics.py
    reports.py
    settings.py
  repositories/
    __init__.py
    user_repository.py
    auth_repository.py
    candidate_repository.py
    note_repository.py
    document_repository.py
    job_repository.py
    pipeline_repository.py
    application_repository.py
    interview_repository.py
    feedback_repository.py
    offer_repository.py
    email_repository.py
    workflow_repository.py
    analytics_repository.py
    report_repository.py
    settings_repository.py
  services/
    __init__.py
    auth_service.py
    user_service.py
    candidate_service.py
    document_service.py
    job_service.py
    pipeline_service.py
    application_service.py
    interview_service.py
    feedback_service.py
    offer_service.py
    email_service.py
    workflow_service.py
    analytics_service.py
    report_service.py
    settings_service.py
  routers/
    __init__.py
    auth.py
    users.py
    candidates.py
    notes.py
    documents.py
    jobs.py
    pipelines.py
    applications.py
    interviews.py
    feedback.py
    offers.py
    emails.py
    workflows.py
    analytics.py
    reports.py
    settings.py

Backend Manifest (JSON)
{
  "project": {
    "name": "talenthub-backend",
    "type": "REST API",
    "runtime": "Python FastAPI",
    "database": "SQLite3",
    "data_layer": "raw SQL with aiosqlite"
  },
  "config": {
    "env_vars": [
      { "name": "DATABASE_PATH", "description": "SQLite database file path", "example": "./data/app.db" },
      { "name": "SECRET_KEY", "description": "JWT signing key", "example": "super-long-random-hex" },
      { "name": "ACCESS_TOKEN_EXPIRES_MINUTES", "description": "Access token TTL (minutes)", "example": "15" },
      { "name": "REFRESH_TOKEN_EXPIRES_DAYS", "description": "Refresh token TTL (days)", "example": "30" },
      { "name": "CORS_ORIGINS", "description": "Comma-separated allowlist", "example": "http://localhost:5173" },
      { "name": "ENV", "description": "Environment (dev|prod)", "example": "dev" },
      { "name": "STORAGE_PATH", "description": "Local file storage root", "example": "./data/storage" },
      { "name": "EMAIL_PROVIDER", "description": "Email provider id (smtp|api)", "example": "smtp" },
      { "name": "EMAIL_FROM", "description": "Default from address", "example": "noreply@talenthub.local" },
      { "name": "ORG_CURRENCY", "description": "Default currency code", "example": "USD" },
      { "name": "RATE_LIMIT_PER_IP_MINUTE", "description": "Optional rate limit toggle/size", "example": "120" }
    ]
  },
  "security": {
    "auth": "JWT in HTTP-only cookies",
    "tokens": {
      "access_ttl_minutes": 15,
      "refresh_ttl_days": 30,
      "refresh_rotation": true,
      "reuse_detection": true
    },
    "csrf": "Double-submit cookie with X-CSRF-Token header for unsafe methods",
    "password_hashing": "argon2id (preferred) or bcrypt",
    "cors_policy": "Allowlist from env with credentials=true",
    "rate_limiting": "Per-IP default + optional per-user in services",
    "rbac": "Role checks in services; roles: admin, recruiter, hiring_manager, interviewer, viewer"
  },
  "api": {
    "base_path": "/api/v1",
    "pagination": { "params": ["page", "page_size"], "envelope": true },
    "errors": { "envelope": { "error": ["code", "message", "details?"] } },
    "idempotency": {
      "create_endpoints": ["POST /candidates/bulk/import", "POST /email/send", "POST /offers", "POST /reports/run"],
      "header": "Idempotency-Key"
    }
  },
  "database": {
    "schema_management": "Execute SQL DDL at startup; CREATE TABLE IF NOT EXISTS",
    "connection_pool": "aiosqlite with connection reuse per request",
    "schema_file": "app/sql/schema.sql",
    "triggers_file": "app/sql/triggers.sql"
  },
  "tables": [
    {
      "name": "organizations",
      "description": "Organizations/tenants",
      "ddl": "CREATE TABLE IF NOT EXISTS organizations ( id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, currency TEXT NOT NULL DEFAULT 'USD', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name);" ]
    },
    {
      "name": "users",
      "description": "User accounts",
      "ddl": "CREATE TABLE IF NOT EXISTS users ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, email TEXT NOT NULL UNIQUE, full_name TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','recruiter','hiring_manager','interviewer','viewer')), password_hash TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);",
        "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);"
      ]
    },
    {
      "name": "refresh_tokens",
      "description": "Refresh token rotation + reuse detection",
      "ddl": "CREATE TABLE IF NOT EXISTS refresh_tokens ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, token_hash TEXT NOT NULL UNIQUE, family_id TEXT NOT NULL, is_revoked INTEGER NOT NULL DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, expires_at DATETIME NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_refresh_family ON refresh_tokens(family_id);",
        "CREATE INDEX IF NOT EXISTS idx_refresh_expires ON refresh_tokens(expires_at);"
      ]
    },
    {
      "name": "candidates",
      "description": "Candidate master records",
      "ddl": "CREATE TABLE IF NOT EXISTS candidates ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, first_name TEXT, last_name TEXT, email TEXT, phone TEXT, location TEXT, source TEXT, rating INTEGER CHECK(rating BETWEEN 1 AND 5), tags TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, deleted_at DATETIME, UNIQUE(org_id, email), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_candidates_org ON candidates(org_id);",
        "CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates(email);",
        "CREATE INDEX IF NOT EXISTS idx_candidates_source ON candidates(source);"
      ]
    },
    {
      "name": "candidate_notes",
      "description": "Notes and communication summaries",
      "ddl": "CREATE TABLE IF NOT EXISTS candidate_notes ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, candidate_id INTEGER NOT NULL, author_user_id INTEGER NOT NULL, body TEXT NOT NULL, visibility TEXT NOT NULL DEFAULT 'internal', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE, FOREIGN KEY(author_user_id) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_cnote_candidate ON candidate_notes(candidate_id);",
        "CREATE INDEX IF NOT EXISTS idx_cnote_org ON candidate_notes(org_id);"
      ]
    },
    {
      "name": "documents",
      "description": "Uploaded candidate documents",
      "ddl": "CREATE TABLE IF NOT EXISTS documents ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, candidate_id INTEGER NOT NULL, kind TEXT CHECK(kind IN ('resume','cover_letter','portfolio','other')) NOT NULL, filename TEXT NOT NULL, content_type TEXT NOT NULL, storage_path TEXT NOT NULL, size_bytes INTEGER NOT NULL, uploaded_by INTEGER, uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE, FOREIGN KEY(uploaded_by) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_docs_candidate ON documents(candidate_id);"
      ]
    },
    {
      "name": "jobs",
      "description": "Job postings",
      "ddl": "CREATE TABLE IF NOT EXISTS jobs ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, title TEXT NOT NULL, department TEXT, location TEXT, employment_type TEXT, description TEXT, status TEXT NOT NULL CHECK(status IN ('draft','open','closed','archived')) DEFAULT 'open', budget_minor INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_jobs_org ON jobs(org_id);",
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);"
      ]
    },
    {
      "name": "pipeline_stages",
      "description": "Per-job pipeline configuration",
      "ddl": "CREATE TABLE IF NOT EXISTS pipeline_stages ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, job_id INTEGER NOT NULL, name TEXT NOT NULL, position INTEGER NOT NULL, is_terminal INTEGER NOT NULL DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(job_id, name), UNIQUE(job_id, position), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_stages_job ON pipeline_stages(job_id);"
      ]
    },
    {
      "name": "applications",
      "description": "Candidate applications to jobs",
      "ddl": "CREATE TABLE IF NOT EXISTS applications ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, candidate_id INTEGER NOT NULL, job_id INTEGER NOT NULL, current_stage_id INTEGER, status TEXT NOT NULL CHECK(status IN ('active','rejected','hired','withdrawn')) DEFAULT 'active', applied_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE, FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE, FOREIGN KEY(current_stage_id) REFERENCES pipeline_stages(id) ON DELETE SET NULL, UNIQUE(candidate_id, job_id) );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_apps_job ON applications(job_id);",
        "CREATE INDEX IF NOT EXISTS idx_apps_candidate ON applications(candidate_id);",
        "CREATE INDEX IF NOT EXISTS idx_apps_status ON applications(status);"
      ]
    },
    {
      "name": "application_events",
      "description": "Stage movements & status changes history",
      "ddl": "CREATE TABLE IF NOT EXISTS application_events ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, application_id INTEGER NOT NULL, event_type TEXT NOT NULL, from_stage_id INTEGER, to_stage_id INTEGER, reason TEXT, actor_user_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_appevents_app ON application_events(application_id);",
        "CREATE INDEX IF NOT EXISTS idx_appevents_type ON application_events(event_type);"
      ]
    },
    {
      "name": "interviews",
      "description": "Interview slots",
      "ddl": "CREATE TABLE IF NOT EXISTS interviews ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, application_id INTEGER NOT NULL, stage_id INTEGER, scheduled_start DATETIME NOT NULL, scheduled_end DATETIME NOT NULL, location TEXT, meeting_link TEXT, calendar_provider TEXT, external_event_id TEXT, status TEXT NOT NULL CHECK(status IN ('scheduled','completed','canceled','no_show')) DEFAULT 'scheduled', created_by INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE, FOREIGN KEY(stage_id) REFERENCES pipeline_stages(id) ON DELETE SET NULL, FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_interviews_app ON interviews(application_id);",
        "CREATE INDEX IF NOT EXISTS idx_interviews_time ON interviews(scheduled_start, scheduled_end);"
      ]
    },
    {
      "name": "interview_participants",
      "description": "Mapping interviewers & attendees",
      "ddl": "CREATE TABLE IF NOT EXISTS interview_participants ( id INTEGER PRIMARY KEY AUTOINCREMENT, interview_id INTEGER NOT NULL, user_id INTEGER, email TEXT, role TEXT CHECK(role IN ('interviewer','observer','candidate')) NOT NULL, UNIQUE(interview_id, user_id, email, role), FOREIGN KEY(interview_id) REFERENCES interviews(id) ON DELETE CASCADE, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_ip_interview ON interview_participants(interview_id);"
      ]
    },
    {
      "name": "feedback_forms",
      "description": "Reusable feedback form templates",
      "ddl": "CREATE TABLE IF NOT EXISTS feedback_forms ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, name TEXT NOT NULL, schema_json TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_ff_org ON feedback_forms(org_id);" ]
    },
    {
      "name": "feedback_responses",
      "description": "Submitted feedback",
      "ddl": "CREATE TABLE IF NOT EXISTS feedback_responses ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, interview_id INTEGER NOT NULL, reviewer_user_id INTEGER NOT NULL, form_id INTEGER, answers_json TEXT NOT NULL, overall_score INTEGER, submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(interview_id, reviewer_user_id), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(interview_id) REFERENCES interviews(id) ON DELETE CASCADE, FOREIGN KEY(reviewer_user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(form_id) REFERENCES feedback_forms(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_fbr_interview ON feedback_responses(interview_id);",
        "CREATE INDEX IF NOT EXISTS idx_fbr_reviewer ON feedback_responses(reviewer_user_id);"
      ]
    },
    {
      "name": "offers",
      "description": "Offers to candidates",
      "ddl": "CREATE TABLE IF NOT EXISTS offers ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, application_id INTEGER NOT NULL, title TEXT NOT NULL, salary_minor INTEGER, currency TEXT NOT NULL, start_date DATE, status TEXT NOT NULL CHECK(status IN ('draft','pending_approval','approved','rejected','sent_to_candidate','accepted','declined','withdrawn')) DEFAULT 'draft', created_by INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE, FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_offers_app ON offers(application_id);",
        "CREATE INDEX IF NOT EXISTS idx_offers_status ON offers(status);"
      ]
    },
    {
      "name": "offer_approvals",
      "description": "Sequential approval chain",
      "ddl": "CREATE TABLE IF NOT EXISTS offer_approvals ( id INTEGER PRIMARY KEY AUTOINCREMENT, offer_id INTEGER NOT NULL, approver_user_id INTEGER NOT NULL, step_order INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')) DEFAULT 'pending', comment TEXT, acted_at DATETIME, UNIQUE(offer_id, approver_user_id), FOREIGN KEY(offer_id) REFERENCES offers(id) ON DELETE CASCADE, FOREIGN KEY(approver_user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_offer_approvals_offer ON offer_approvals(offer_id);"
      ]
    },
    {
      "name": "email_templates",
      "description": "Templated emails with variables",
      "ddl": "CREATE TABLE IF NOT EXISTS email_templates ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, name TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(org_id, name), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_et_org ON email_templates(org_id);" ]
    },
    {
      "name": "email_sequences",
      "description": "Automated multi-step sequences",
      "ddl": "CREATE TABLE IF NOT EXISTS email_sequences ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, name TEXT NOT NULL, description TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(org_id, name), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_es_org ON email_sequences(org_id);" ]
    },
    {
      "name": "email_sequence_steps",
      "description": "Steps within sequences",
      "ddl": "CREATE TABLE IF NOT EXISTS email_sequence_steps ( id INTEGER PRIMARY KEY AUTOINCREMENT, sequence_id INTEGER NOT NULL, step_order INTEGER NOT NULL, template_id INTEGER NOT NULL, delay_hours INTEGER NOT NULL DEFAULT 0, FOREIGN KEY(sequence_id) REFERENCES email_sequences(id) ON DELETE CASCADE, FOREIGN KEY(template_id) REFERENCES email_templates(id) ON DELETE CASCADE, UNIQUE(sequence_id, step_order) );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_ess_seq ON email_sequence_steps(sequence_id);" ]
    },
    {
      "name": "email_queue",
      "description": "Outbox to send emails",
      "ddl": "CREATE TABLE IF NOT EXISTS email_queue ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, candidate_id INTEGER, to_email TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('queued','sending','sent','failed','canceled')) DEFAULT 'queued', error TEXT, scheduled_at DATETIME DEFAULT CURRENT_TIMESTAMP, sent_at DATETIME, idempotency_key TEXT, UNIQUE(idempotency_key), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE SET NULL );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_eq_status ON email_queue(status);",
        "CREATE INDEX IF NOT EXISTS idx_eq_scheduled ON email_queue(scheduled_at);"
      ]
    },
    {
      "name": "email_events",
      "description": "Provider delivery/open/click events",
      "ddl": "CREATE TABLE IF NOT EXISTS email_events ( id INTEGER PRIMARY KEY AUTOINCREMENT, queue_id INTEGER NOT NULL, event_type TEXT NOT NULL, payload_json TEXT, occurred_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(queue_id) REFERENCES email_queue(id) ON DELETE CASCADE );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_eevents_queue ON email_events(queue_id);" ]
    },
    {
      "name": "workflow_rules",
      "description": "Custom automation rules",
      "ddl": "CREATE TABLE IF NOT EXISTS workflow_rules ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, name TEXT NOT NULL, is_enabled INTEGER NOT NULL DEFAULT 1, trigger TEXT NOT NULL CHECK(trigger IN ('on_stage_enter','on_stage_exit','on_status_change')), condition_json TEXT, action_json TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_wf_org ON workflow_rules(org_id);" ]
    },
    {
      "name": "sources",
      "description": "Candidate/job sources & costs",
      "ddl": "CREATE TABLE IF NOT EXISTS sources ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, name TEXT NOT NULL, monthly_cost_minor INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(org_id, name), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_sources_org ON sources(org_id);" ]
    },
    {
      "name": "candidate_sources",
      "description": "Link candidates/applications to a source",
      "ddl": "CREATE TABLE IF NOT EXISTS candidate_sources ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, candidate_id INTEGER NOT NULL, source_id INTEGER NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(candidate_id, source_id), FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE, FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE );",
      "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_csrc_candidate ON candidate_sources(candidate_id);",
        "CREATE INDEX IF NOT EXISTS idx_csrc_source ON candidate_sources(source_id);"
      ]
    },
    {
      "name": "imports",
      "description": "Bulk import jobs",
      "ddl": "CREATE TABLE IF NOT EXISTS imports ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, created_by INTEGER NOT NULL, filename TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','processing','completed','failed')) DEFAULT 'pending', total_rows INTEGER DEFAULT 0, processed_rows INTEGER DEFAULT 0, error TEXT, idempotency_key TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE CASCADE, UNIQUE(idempotency_key) );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_imports_status ON imports(status);" ]
    },
    {
      "name": "reports",
      "description": "Report runs & exports",
      "ddl": "CREATE TABLE IF NOT EXISTS reports ( id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL, kind TEXT NOT NULL, params_json TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')) DEFAULT 'completed', rows INTEGER DEFAULT 0, storage_path TEXT, created_by INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE, FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": [ "CREATE INDEX IF NOT EXISTS idx_reports_org ON reports(org_id);" ]
    }
  ],
  "triggers": {
    "updated_at": [
      "CREATE TRIGGER IF NOT EXISTS trg_users_updated_at AFTER UPDATE ON users BEGIN UPDATE users SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;",
      "CREATE TRIGGER IF NOT EXISTS trg_jobs_updated_at AFTER UPDATE ON jobs BEGIN UPDATE jobs SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;",
      "CREATE TRIGGER IF NOT EXISTS trg_applications_updated_at AFTER UPDATE ON applications BEGIN UPDATE applications SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;",
      "CREATE TRIGGER IF NOT EXISTS trg_interviews_updated_at AFTER UPDATE ON interviews BEGIN UPDATE interviews SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;",
      "CREATE TRIGGER IF NOT EXISTS trg_offers_updated_at AFTER UPDATE ON offers BEGIN UPDATE offers SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;"
    ]
  },
  "entities": [
    { "name": "User", "table": "users", "fields": { "id": "int", "org_id": "int", "email": "str", "full_name": "str", "role": "enum", "is_active": "bool" } },
    { "name": "Candidate", "table": "candidates", "fields": { "id": "int", "org_id": "int", "name": "first/last", "email": "str", "phone": "str", "source": "str", "rating": "int" } },
    { "name": "Job", "table": "jobs", "fields": { "id": "int", "org_id": "int", "title": "str", "status": "enum", "budget_minor": "int" } },
    { "name": "PipelineStage", "table": "pipeline_stages", "fields": { "id": "int", "job_id": "int", "name": "str", "position": "int" } },
    { "name": "Application", "table": "applications", "fields": { "id": "int", "candidate_id": "int", "job_id": "int", "current_stage_id": "int", "status": "enum" } },
    { "name": "Interview", "table": "interviews", "fields": { "id": "int", "application_id": "int", "scheduled_start": "datetime", "status": "enum" } },
    { "name": "Feedback", "table": "feedback_responses", "fields": { "id": "int", "interview_id": "int", "reviewer_user_id": "int", "overall_score": "int" } },
    { "name": "Offer", "table": "offers", "fields": { "id": "int", "application_id": "int", "status": "enum", "salary_minor": "int" } },
    { "name": "EmailTemplate", "table": "email_templates", "fields": { "id": "int", "name": "str" } },
    { "name": "EmailSequence", "table": "email_sequences", "fields": { "id": "int", "name": "str" } },
    { "name": "WorkflowRule", "table": "workflow_rules", "fields": { "id": "int", "trigger": "enum", "condition_json": "json", "action_json": "json" } }
  ],
  "schemas": {
    "auth": ["AuthRegisterRequest", "AuthLoginRequest", "AuthLoginResponse", "AuthRefreshResponse", "CsrfTokenResponse", "UserResponse", "ChangePasswordRequest"],
    "users": ["UserCreateRequest", "UserUpdateRequest", "PaginatedUserResponse"],
    "candidates": ["CandidateCreateRequest", "CandidateUpdateRequest", "CandidateResponse", "CandidateDetailResponse", "PaginatedCandidateResponse", "CandidateScoreRequest", "CandidateScoreResponse", "ImportJobResponse", "ImportJobDetailResponse"],
    "notes": ["NoteCreateRequest", "NoteResponse", "PaginatedNoteResponse"],
    "documents": ["DocumentResponse", "PaginatedDocumentResponse"],
    "jobs": ["JobCreateRequest", "JobUpdateRequest", "JobResponse", "JobDetailResponse", "PaginatedJobResponse"],
    "pipelines": ["PipelineStageCreateRequest", "PipelineStageResponse", "PipelineStageListResponse"],
    "applications": ["ApplicationCreateRequest", "ApplicationMoveRequest", "ApplicationRejectRequest", "ApplicationResponse", "ApplicationDetailResponse", "PaginatedApplicationResponse"],
    "interviews": ["InterviewCreateRequest", "InterviewResponse", "PaginatedInterviewResponse"],
    "feedback": ["FeedbackFormResponse", "FeedbackSubmitRequest", "FeedbackResponse", "PaginatedFeedbackResponse"],
    "offers": ["OfferCreateRequest", "OfferResponse", "OfferDetailResponse", "PaginatedOfferResponse", "OfferApproveRequest"],
    "emails": ["EmailTemplateCreateRequest", "EmailTemplateResponse", "PaginatedEmailTemplateResponse", "EmailSequenceCreateRequest", "EmailSequenceResponse", "PaginatedEmailSequenceResponse", "EmailSendRequest", "EmailQueueResponse"],
    "workflows": ["WorkflowRuleCreateRequest", "WorkflowRuleResponse", "PaginatedWorkflowRuleResponse"],
    "analytics": ["FunnelAnalyticsResponse", "TimeToHireResponse", "SourceEffectivenessResponse", "HiringManagerPerformanceResponse"],
    "reports": ["ReportRunRequest", "ReportResultResponse"],
    "settings": ["SettingsResponse", "SettingsUpdateRequest"],
    "common": ["ErrorResponse", "SuccessResponse", "HealthResponse"]
  },
  "repositories": [
    { "name": "UserRepository", "file": "app/repositories/user_repository.py", "operations": [
      { "method": "create", "sql": "INSERT INTO users (org_id, email, full_name, role, password_hash) VALUES (?, ?, ?, ?, ?)" },
      { "method": "find_by_email", "sql": "SELECT * FROM users WHERE email = ?" },
      { "method": "get", "sql": "SELECT * FROM users WHERE id = ? AND org_id = ?" },
      { "method": "list", "sql": "SELECT * FROM users WHERE org_id = ? LIMIT ? OFFSET ?" },
      { "method": "update", "sql": "UPDATE users SET full_name=?, role=?, is_active=? WHERE id=? AND org_id=?" }
    ]},
    { "name": "CandidateRepository", "file": "app/repositories/candidate_repository.py", "operations": [
      { "method": "create", "sql": "INSERT INTO candidates (org_id, first_name, last_name, email, phone, location, source, rating, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)" },
      { "method": "get", "sql": "SELECT * FROM candidates WHERE id=? AND org_id=?" },
      { "method": "update", "sql": "UPDATE candidates SET first_name=?, last_name=?, email=?, phone=?, location=?, source=?, rating=?, tags=? WHERE id=? AND org_id=?" },
      { "method": "list", "sql": "SELECT * FROM candidates WHERE org_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?" }
    ]},
    { "name": "JobRepository", "file": "app/repositories/job_repository.py", "operations": [
      { "method": "create", "sql": "INSERT INTO jobs (org_id, title, department, location, employment_type, description, status, budget_minor) VALUES (?, ?, ?, ?, ?, ?, ?, ?)" },
      { "method": "get", "sql": "SELECT * FROM jobs WHERE id=? AND org_id=?" },
      { "method": "update", "sql": "UPDATE jobs SET title=?, department=?, location=?, employment_type=?, description=?, status=?, budget_minor=? WHERE id=? AND org_id=?" },
      { "method": "list", "sql": "SELECT * FROM jobs WHERE org_id=? AND status != 'archived' LIMIT ? OFFSET ?" }
    ]},
    { "name": "PipelineRepository", "file": "app/repositories/pipeline_repository.py", "operations": [
      { "method": "list_stages", "sql": "SELECT * FROM pipeline_stages WHERE job_id=? ORDER BY position" },
      { "method": "create_stage", "sql": "INSERT INTO pipeline_stages (org_id, job_id, name, position, is_terminal) VALUES (?, ?, ?, ?, ?)" },
      { "method": "update_stage", "sql": "UPDATE pipeline_stages SET name=?, position=?, is_terminal=? WHERE id=? AND job_id=?" },
      { "method": "delete_stage", "sql": "DELETE FROM pipeline_stages WHERE id=? AND job_id=?" }
    ]},
    { "name": "ApplicationRepository", "file": "app/repositories/application_repository.py", "operations": [
      { "method": "create", "sql": "INSERT INTO applications (org_id, candidate_id, job_id, current_stage_id, status) VALUES (?, ?, ?, ?, 'active')" },
      { "method": "get", "sql": "SELECT * FROM applications WHERE id=? AND org_id=?" },
      { "method": "update_stage", "sql": "UPDATE applications SET current_stage_id=? WHERE id=? AND org_id=?" },
      { "method": "update_status", "sql": "UPDATE applications SET status=? WHERE id=? AND org_id=?" },
      { "method": "log_event", "sql": "INSERT INTO application_events (org_id, application_id, event_type, from_stage_id, to_stage_id, reason, actor_user_id) VALUES (?, ?, ?, ?, ?, ?, ?)" }
    ]},
    { "name": "InterviewRepository", "file": "app/repositories/interview_repository.py", "operations": [
      { "method": "create", "sql": "INSERT INTO interviews (org_id, application_id, stage_id, scheduled_start, scheduled_end, location, meeting_link, calendar_provider, external_event_id, status, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?)" },
      { "method": "list_for_app", "sql": "SELECT * FROM interviews WHERE application_id=? ORDER BY scheduled_start DESC" },
      { "method": "update", "sql": "UPDATE interviews SET scheduled_start=?, scheduled_end=?, location=?, meeting_link=?, status=? WHERE id=? AND org_id=?" },
      { "method": "add_participant", "sql": "INSERT INTO interview_participants (interview_id, user_id, email, role) VALUES (?, ?, ?, ?)" }
    ]},
    { "name": "FeedbackRepository", "file": "app/repositories/feedback_repository.py", "operations": [
      { "method": "submit", "sql": "INSERT INTO feedback_responses (org_id, interview_id, reviewer_user_id, form_id, answers_json, overall_score) VALUES (?, ?, ?, ?, ?, ?)" },
      { "method": "list_for_interview", "sql": "SELECT * FROM feedback_responses WHERE interview_id=?" }
    ]},
    { "name": "OfferRepository", "file": "app/repositories/offer_repository.py", "operations": [
      { "method": "create", "sql": "INSERT INTO offers (org_id, application_id, title, salary_minor, currency, start_date, status, created_by) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?)" },
      { "method": "get", "sql": "SELECT * FROM offers WHERE id=? AND org_id=?" },
      { "method": "update_status", "sql": "UPDATE offers SET status=? WHERE id=? AND org_id=?" },
      { "method": "insert_approver", "sql": "INSERT INTO offer_approvals (offer_id, approver_user_id, step_order) VALUES (?, ?, ?)" },
      { "method": "set_approver_status", "sql": "UPDATE offer_approvals SET status=?, comment=?, acted_at=CURRENT_TIMESTAMP WHERE offer_id=? AND approver_user_id=?" }
    ]},
    { "name": "EmailRepository", "file": "app/repositories/email_repository.py", "operations": [
      { "method": "create_template", "sql": "INSERT INTO email_templates (org_id, name, subject, body) VALUES (?, ?, ?, ?)" },
      { "method": "enqueue", "sql": "INSERT INTO email_queue (org_id, candidate_id, to_email, subject, body, status, idempotency_key) VALUES (?, ?, ?, ?, ?, 'queued', ?)" },
      { "method": "mark_sent", "sql": "UPDATE email_queue SET status='sent', sent_at=CURRENT_TIMESTAMP WHERE id=?" }
    ]},
    { "name": "WorkflowRepository", "file": "app/repositories/workflow_repository.py", "operations": [
      { "method": "create_rule", "sql": "INSERT INTO workflow_rules (org_id, name, is_enabled, trigger, condition_json, action_json) VALUES (?, ?, ?, ?, ?, ?)" },
      { "method": "list_enabled", "sql": "SELECT * FROM workflow_rules WHERE org_id=? AND is_enabled=1" }
    ]},
    { "name": "AnalyticsRepository", "file": "app/repositories/analytics_repository.py", "operations": [
      { "method": "funnel", "sql": "-- SELECT stage conversions via application_events" },
      { "method": "time_to_hire", "sql": "-- SELECT date diffs from applied to hired" },
      { "method": "source_effectiveness", "sql": "-- JOIN candidate_sources, applications, offers" }
    ]}
  ],
  "services": [
    { "name": "AuthService", "depends_on": ["AuthRepository", "UserRepository"], "responsibilities": ["login", "register", "token rotation", "logout"] },
    { "name": "CandidateService", "depends_on": ["CandidateRepository", "DocumentService", "EmailService"], "responsibilities": ["CRUD", "scoring", "import"] },
    { "name": "JobService", "depends_on": ["JobRepository", "PipelineRepository"], "responsibilities": ["CRUD jobs", "manage stages"] },
    { "name": "ApplicationService", "depends_on": ["ApplicationRepository", "WorkflowService", "EmailService"], "responsibilities": ["create/move/reject", "history"] },
    { "name": "InterviewService", "depends_on": ["InterviewRepository"], "responsibilities": ["schedule/reschedule/cancel", "conflict detection"] },
    { "name": "FeedbackService", "depends_on": ["FeedbackRepository"], "responsibilities": ["forms", "submit/list feedback"] },
    { "name": "OfferService", "depends_on": ["OfferRepository", "EmailService"], "responsibilities": ["create/submit/approve/reject/withdraw"] },
    { "name": "EmailService", "depends_on": ["EmailRepository"], "responsibilities": ["render templates", "enqueue", "sequence runner"] },
    { "name": "WorkflowService", "depends_on": ["WorkflowRepository", "EmailService", "ApplicationRepository"], "responsibilities": ["evaluate rules on events", "perform actions"] },
    { "name": "AnalyticsService", "depends_on": ["AnalyticsRepository"], "responsibilities": ["funnel", "time-to-hire", "source effectiveness"] },
    { "name": "ReportService", "depends_on": ["AnalyticsRepository", "ReportRepository"], "responsibilities": ["build CSV from query", "persist export"] },
    { "name": "SettingsService", "depends_on": ["SettingsRepository"], "responsibilities": ["read/update org settings"] }
  ],
  "routers": [
    { "file": "app/routers/auth.py", "tag": "Auth", "endpoints": ["/auth/register","/auth/login","/auth/refresh","/auth/logout","/auth/me","/auth/csrf","/auth/change-password"] },
    { "file": "app/routers/users.py", "tag": "Users", "endpoints": ["/users", "/users/{user_id}"] },
    { "file": "app/routers/candidates.py", "tag": "Candidates", "endpoints": ["/candidates","/candidates/{candidate_id}","/candidates/{candidate_id}/score","/candidates/bulk/import","/candidates/imports/{job_id}"] },
    { "file": "app/routers/notes.py", "tag": "Candidate Notes", "endpoints": ["/candidates/{candidate_id}/notes"] },
    { "file": "app/routers/documents.py", "tag": "Documents", "endpoints": ["/candidates/{candidate_id}/documents"] },
    { "file": "app/routers/jobs.py", "tag": "Jobs", "endpoints": ["/jobs","/jobs/{job_id}"] },
    { "file": "app/routers/pipelines.py", "tag": "Pipelines", "endpoints": ["/jobs/{job_id}/stages"] },
    { "file": "app/routers/applications.py", "tag": "Applications", "endpoints": ["/applications","/applications/{app_id}","/applications/{app_id}/move","/applications/{app_id}/reject"] },
    { "file": "app/routers/interviews.py", "tag": "Interviews", "endpoints": ["/interviews","/interviews/{interview_id}","/interviews/{interview_id}/feedback","/interviews/{interview_id}/feedback-form"] },
    { "file": "app/routers/offers.py", "tag": "Offers", "endpoints": ["/offers","/offers/{offer_id}","/offers/{offer_id}/submit","/offers/{offer_id}/approve","/offers/{offer_id}/reject","/offers/{offer_id}/withdraw"] },
    { "file": "app/routers/emails.py", "tag": "Emails", "endpoints": ["/email/templates","/email/sequences","/email/send"] },
    { "file": "app/routers/workflows.py", "tag": "Workflows", "endpoints": ["/workflows/rules","/workflows/rules/{rule_id}"] },
    { "file": "app/routers/analytics.py", "tag": "Analytics", "endpoints": ["/analytics/funnel","/analytics/time-to-hire","/analytics/source-effectiveness","/analytics/hiring-manager-performance"] },
    { "file": "app/routers/reports.py", "tag": "Reports", "endpoints": ["/reports/run","/reports/{report_id}/export"] },
    { "file": "app/routers/settings.py", "tag": "Settings", "endpoints": ["/settings","/health"] }
  ],
  "endpoints": [
    {
      "method": "POST",
      "path": "/api/v1/applications/{app_id}/move",
      "summary": "Move application to another stage",
      "description": "Transitions an application between pipeline stages, logs event, evaluates workflow rules, and may enqueue emails.",
      "authentication": "authenticated (recruiter or hiring_manager)",
      "path_parameters": [
        { "name": "app_id", "type": "integer", "validation": "positive" }
      ],
      "query_parameters": [],
      "request_body": {
        "schema": "ApplicationMoveRequest",
        "required": true,
        "example": { "to_stage_id": 12, "note": "Proceed to onsite" }
      },
      "responses": {
        "200": { "description": "Moved successfully", "schema": "ApplicationResponse" },
        "400": { "description": "Invalid transition", "schema": "ErrorResponse" },
        "404": { "description": "Application or stage not found", "schema": "ErrorResponse" }
      },
      "http_status": ["200","400","404"],
      "rate_limit": "60/min per user"
    },
    {
      "method": "POST",
      "path": "/api/v1/interviews",
      "summary": "Schedule an interview",
      "description": "Schedules interview with conflict detection against existing interviews for participants.",
      "authentication": "authenticated (recruiter or hiring_manager)",
      "path_parameters": [],
      "query_parameters": [],
      "request_body": {
        "schema": "InterviewCreateRequest",
        "required": true,
        "example": {
          "application_id": 101,
          "stage_id": 12,
          "scheduled_start": "2025-08-20T09:00:00Z",
          "scheduled_end": "2025-08-20T10:00:00Z",
          "participants": [{ "user_id": 7, "role": "interviewer" }]
        }
      },
      "responses": {
        "201": { "description": "Interview created", "schema": "InterviewResponse" },
        "409": { "description": "Conflicts found", "schema": "ErrorResponse", "example": { "error": { "code": "CONFLICT", "message": "Interviewer busy", "details": [{ "user_id": 7, "overlap": true }] } } }
      },
      "http_status": ["201","409"],
      "rate_limit": "30/min per user"
    }
  ],
  "middleware": [
    "CORS (origins from env, allow credentials)",
    "Request ID injection and propagation",
    "Structured logging (JSON) with latency/route/status",
    "Auth cookie parser",
    "CSRF validator for unsafe methods",
    "Optional rate limiter (per-IP) from env"
  ],
  "observability": {
    "logging": "JSON lines with request_id, method, path, status, duration_ms, user_id?",
    "request_id": "Generated per request, echoed in responses and logs"
  }
}



This is documentation for pydantic 2:
Pydantic V1	Pydantic V2
pydantic.BaseSettings	pydantic_settings.BaseSettings
pydantic.color	pydantic_extra_types.color
pydantic.types.PaymentCardBrand	pydantic_extra_types.PaymentCardBrand
pydantic.types.PaymentCardNumber	pydantic_extra_types.PaymentCardNumber
pydantic.utils.version_info	pydantic.version.version_info
pydantic.error_wrappers.ValidationError	pydantic.ValidationError
pydantic.utils.to_camel	pydantic.alias_generators.to_pascal
pydantic.utils.to_lower_camel	pydantic.alias_generators.to_camel
pydantic.PyObject	pydantic.ImportString
Deprecated and moved in Pydantic V2¶
Pydantic V1	Pydantic V2
pydantic.tools.schema_of	pydantic.deprecated.tools.schema_of
pydantic.tools.parse_obj_as	pydantic.deprecated.tools.parse_obj_as
pydantic.tools.schema_json_of	pydantic.deprecated.tools.schema_json_of
pydantic.json.pydantic_encoder	pydantic.deprecated.json.pydantic_encoder
pydantic.validate_arguments	pydantic.deprecated.decorator.validate_arguments
pydantic.json.custom_pydantic_encoder	pydantic.deprecated.json.custom_pydantic_encoder
pydantic.json.ENCODERS_BY_TYPE	pydantic.deprecated.json.ENCODERS_BY_TYPE
pydantic.json.timedelta_isoformat	pydantic.deprecated.json.timedelta_isoformat
pydantic.decorator.validate_arguments	pydantic.deprecated.decorator.validate_arguments
pydantic.class_validators.validator	pydantic.deprecated.class_validators.validator
pydantic.class_validators.root_validator	pydantic.deprecated.class_validators.root_validator
pydantic.utils.deep_update	pydantic.v1.utils.deep_update
pydantic.utils.GetterDict	pydantic.v1.utils.GetterDict
pydantic.utils.lenient_issubclass	pydantic.v1.utils.lenient_issubclass
pydantic.utils.lenient_isinstance	pydantic.v1.utils.lenient_isinstance
pydantic.utils.is_valid_field	pydantic.v1.utils.is_valid_field
pydantic.utils.update_not_none	pydantic.v1.utils.update_not_none
pydantic.utils.import_string	pydantic.v1.utils.import_string
pydantic.utils.Representation	pydantic.v1.utils.Representation
pydantic.utils.ROOT_KEY	pydantic.v1.utils.ROOT_KEY
pydantic.utils.smart_deepcopy	pydantic.v1.utils.smart_deepcopy
pydantic.utils.sequence_like	pydantic.v1.utils.sequence_like

Changes to pydantic.BaseModel¶
Various method names have been changed; all non-deprecated BaseModel methods now have names matching either the format model_.* or __.*pydantic.*__. Where possible, we have retained the deprecated methods with their old names to help ease migration, but calling them will emit DeprecationWarnings.

Pydantic V1	Pydantic V2
__fields__	model_fields
__private_attributes__	__pydantic_private__
__validators__	__pydantic_validator__
construct()	model_construct()
copy()	model_copy()
dict()	model_dump()
json_schema()	model_json_schema()
json()	model_dump_json()
parse_obj()	model_validate()
update_forward_refs()	model_rebuild()


The test implementation must follow simple fastapi test. test each functionaality in its own test in a single file:
Assume Pydantic ≥2 and use only v2 names/APIs. Use model_config = ConfigDict(...) on the class (do not define class Config). Typical flags you'll set: from_attributes=True, populate_by_name=True, extra='forbid'|'ignore'|'allow', strict=True. Avoid field names starting with model_ unless you change protected_namespaces. 


Methods were renamed: .dict()→model_dump(), .json()→model_dump_json(), parse_obj()→model_validate(), parse_raw()→model_validate_json(), construct()→model_construct(), json_schema()→model_json_schema(). If you previously used from_orm, set model_config = ConfigDict(from_attributes=True) and call model_validate. 


Validation: replace @validator with @field_validator for per-field checks and coercion; replace @root_validator with @model_validator(mode='before'|'after') for cross-field logic. Use the documented signatures (e.g., (cls, v, info: FieldValidationInfo) for field validators). 



Serialization customization is decorator-based: use @field_serializer/@model_serializer; add read-only derived values with @computed_field so they appear in dumps. 


Constraints & types: prefer typing.Annotated with Field(...) or StringConstraints(...) instead of con* helpers like constr/conint (discouraged and slated for deprecation). Example: name: Annotated[str, StringConstraints(min_length=1)]. 


Ad-hoc (non-model) validation/serialization: use TypeAdapter(T) (e.g., TypeAdapter(list[int]).validate_python(data)), not parse_obj_as. 


Single-value/root models: use RootModel[T] (v1's __root__ is replaced). 

Settings: import from pydantic_settings (from pydantic_settings import BaseSettings, SettingsConfigDict) and configure with model_config = SettingsConfigDict(...) (e.g., env_prefix='APP_', env_file='.env')

This is documentation for pydantic 2:
Pydantic V1	Pydantic V2
pydantic.BaseSettings	pydantic_settings.BaseSettings
pydantic.color	pydantic_extra_types.color
pydantic.types.PaymentCardBrand	pydantic_extra_types.PaymentCardBrand
pydantic.types.PaymentCardNumber	pydantic_extra_types.PaymentCardNumber
pydantic.utils.version_info	pydantic.version.version_info
pydantic.error_wrappers.ValidationError	pydantic.ValidationError
pydantic.utils.to_camel	pydantic.alias_generators.to_pascal
pydantic.utils.to_lower_camel	pydantic.alias_generators.to_camel
pydantic.PyObject	pydantic.ImportString
Deprecated and moved in Pydantic V2¶
Pydantic V1	Pydantic V2
pydantic.tools.schema_of	pydantic.deprecated.tools.schema_of
pydantic.tools.parse_obj_as	pydantic.deprecated.tools.parse_obj_as
pydantic.tools.schema_json_of	pydantic.deprecated.tools.schema_json_of
pydantic.json.pydantic_encoder	pydantic.deprecated.json.pydantic_encoder
pydantic.validate_arguments	pydantic.deprecated.decorator.validate_arguments
pydantic.json.custom_pydantic_encoder	pydantic.deprecated.json.custom_pydantic_encoder
pydantic.json.ENCODERS_BY_TYPE	pydantic.deprecated.json.ENCODERS_BY_TYPE
pydantic.json.timedelta_isoformat	pydantic.deprecated.json.timedelta_isoformat
pydantic.decorator.validate_arguments	pydantic.deprecated.decorator.validate_arguments
pydantic.class_validators.validator	pydantic.deprecated.class_validators.validator
pydantic.class_validators.root_validator	pydantic.deprecated.class_validators.root_validator
pydantic.utils.deep_update	pydantic.v1.utils.deep_update
pydantic.utils.GetterDict	pydantic.v1.utils.GetterDict
pydantic.utils.lenient_issubclass	pydantic.v1.utils.lenient_issubclass
pydantic.utils.lenient_isinstance	pydantic.v1.utils.lenient_isinstance
pydantic.utils.is_valid_field	pydantic.v1.utils.is_valid_field
pydantic.utils.update_not_none	pydantic.v1.utils.update_not_none
pydantic.utils.import_string	pydantic.v1.utils.import_string
pydantic.utils.Representation	pydantic.v1.utils.Representation
pydantic.utils.ROOT_KEY	pydantic.v1.utils.ROOT_KEY
pydantic.utils.smart_deepcopy	pydantic.v1.utils.smart_deepcopy
pydantic.utils.sequence_like	pydantic.v1.utils.sequence_like

Changes to pydantic.BaseModel¶
Various method names have been changed; all non-deprecated BaseModel methods now have names matching either the format model_.* or __.*pydantic.*__. Where possible, we have retained the deprecated methods with their old names to help ease migration, but calling them will emit DeprecationWarnings.

Pydantic V1	Pydantic V2
__fields__	model_fields
__private_attributes__	__pydantic_private__
__validators__	__pydantic_validator__
construct()	model_construct()
copy()	model_copy()
dict()	model_dump()
json_schema()	model_json_schema()
json()	model_dump_json()
parse_obj()	model_validate()
update_forward_refs()	model_rebuild()


The test implementation must follow simple fastapi test. test each functionaality in its own test in a single file:
Here's the FastAPI testing format:
Basic Structure:
pythonfrom fastapi.testclient import TestClient
from .main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_function_name(client):
    response = client.get("/endpoint")
    assert response.status_code == 200
    assert response.json() == expected_data

dont forget to add database file to create the actual database in a file. be careful that the tests all need a clean database. Follow me given file structure carefully. Be careful not to run into sqlite3.OperationalError: database is locked. These are your allowed commands: ALLOWED_COMMANDS = {
    "node", "npm", "pnpm", "yarn", "npx", "vite",
    "python", "pip", "uv", "pytest", "ruff", "black", "mypy",
    "esbuild", "tsc", "git",
}

Prefer to read only function definitions of file with the according tool. Only if you need the content of the file read it. And again its PYDANTIC 2.0.0 and above DO NOT USE OLD FEATURES.
    """
    
    llm = ChatOpenAI(
        model="gpt-5-mini", 
        max_retries=15
    )
    test_llm = ChatOpenAI(
        model="gpt-5-mini", 
        max_retries=15
    )
    planning_llm = ChatOpenAI(
        model="gpt-5", 
        max_retries=15
    )
    
    result = mcp_fast_iterative(llm, test_llm, planning_llm, "./workspace3", message, batch_size=10)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    planning = result["planning_phase"]
    implementation = result["implementation_phase"]
    testing = result.get("testing_phase", {})
    
    print(f"\nPlanning Phase:")
    print(f"- Files planned: {len(planning['plan']['files'])}")
    
    print(f"\nImplementation Phase:")
    print(f"- Files successfully implemented: {implementation['files_implemented']}")
    print(f"- Files failed: {implementation['files_failed']}")
    
    if implementation['files_failed'] > 0:
        print("\nFailed files:")
        for res in implementation['results']:
            if res['status'] == 'error':
                print(f"- {res['file']}: {res['error']}")
    
    print(f"\nTesting Phase:")
    print(f"- Status: {testing.get('status', 'N/A')}")
    if testing.get('status') == 'error':
        print(f"- Error: {testing.get('error')}")
    else:
        print(f"- Final Message: {testing.get('final_message', 'No message.')[:300]}...")

    print(f"\nImplementation completed!")
