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
"""

TESTING_SYSTEM_PROMPT = """\
You are a software quality assurance engineer. Your task is to test the implemented code by creating and running a comprehensive test suite.

INSTRUCTIONS:
1. Create a `test_backend.py` file to test all API endpoints.
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

def mcp_fast_iterative(model: BaseChatModel, workspace_root: Union[str, Path], project_description: str, batch_size: int = 1) -> Dict[str, Any]:
    """
    Fast implementation using a three-phase approach with batched implementation.
    1. Planning phase: Create implementation plan.
    2. Implementation phase: Iterate through files in batches of N.
    3. Testing phase: Create and run tests until they all pass.
    """
    print("=== PHASE 1: PLANNING ===")
    
    planning_agent = build_planning_agent(model)
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
    
    testing_agent = build_testing_agent(model, tools)
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
                      Implement this backend. It must be runnable without errors. If you use pydantic remember to use pydantic-settings and pydantic bigger than 2.0.0 as this will be installed. dont use old syntax. Also create a test_backend.py file that tests the backend. Make it production ready dont simplify. Run the test after impleemntation and fix until it works. Make test suite test most functionality. if no registration endpoint is there add it as well. Test after implementing:
                     Assumptions

Domain & Region: Global storefront; default currency USD; prices stored as integer minor units (cents). Taxes are simple rate-per-order (e.g., provided by frontend config or a simple server rule), no complex nexus logic.

Payments: Cards via Stripe Payment Intents; PayPal via Orders API. Webhooks supported for both. (Env keys provided; sandbox/live via ENV.)

Email: SMTP provider (e.g., SendGrid / Postmark SMTP). All transactional emails sent via background tasks within the API process (simple queue with asyncio tasks).

Images: Stored on disk at ./data/uploads and served via authenticated signed URLs; in production, replace with object storage (S3-compatible).

Shipping: Flat-rate options (standard, express) with static costs; shipping labels are generated as PDFs with our own template (data-only endpoint here).

Addresses: Basic validation only (format & required fields). No external address verification.

RBAC: Roles: customer, admin, support. Admin dashboard requires admin; support tickets require support or admin.

Account Lockout: After 5 failed login attempts in 15 minutes, lock for 15 minutes; unlock link via email allowed.

"Remember me": Extends refresh token TTL from 7 days to 30 days (cookie persistent).

Catalog: Chairs only. Variants by color and material, optional finish. Inventory tracked per SKU (variant).

Guest checkout: Allowed; orders store guest_email and ephemeral addresses. If guest later registers with same email, history can be linked manually by admin.

Pagination defaults: page=1, page_size=20 (max 100).

Sorting allowlist: price, popularity, created_at.

Search: SQLite FTS5 virtual table for product search by title/description; if unavailable at runtime, fallbacks to LIKE.

Rates limiting: Per-IP for public endpoints; per-user for authenticated. Suggested values appear below.

Backend API Plan

Listed as METHOD /api/v1/... — purpose (+ auth, request/response schemas)

Auth & Account

POST /api/v1/auth/register — Register with email/username/password; send verification email. (+ public; UserRegisterRequest → UserResponse)

POST /api/v1/auth/login — Login, set HTTP-only cookies (access/refresh), "remember me". (+ public; LoginRequest → AuthSessionResponse)

POST /api/v1/auth/logout — Logout, revoke refresh token. (+ auth optional; clears cookies; Empty → OKResponse)

POST /api/v1/auth/refresh — Rotate refresh, issue new tokens. (+ cookie-based; Empty → AuthSessionResponse)

POST /api/v1/auth/verify-email/request — Send verification email. (+ auth; Empty → OKResponse)

POST /api/v1/auth/verify-email/confirm — Confirm via token. (+ public; EmailVerifyConfirmRequest → OKResponse)

POST /api/v1/auth/password/forgot — Send reset email. (+ public; PasswordResetRequest → OKResponse)

POST /api/v1/auth/password/reset — Reset via token. (+ public; PasswordResetConfirmRequest → OKResponse)

GET /api/v1/auth/csrf — Issue CSRF token (double-submit). (+ public; Empty → CsrfTokenResponse)

Profile & Addresses

GET /api/v1/me — Current user profile. (+ auth; Empty → UserResponse)

PUT /api/v1/me — Update profile fields. (+ auth+CSRF; UserUpdateRequest → UserResponse)

PUT /api/v1/me/password — Change password. (+ auth+CSRF; ChangePasswordRequest → OKResponse)

GET /api/v1/me/addresses — List addresses. (+ auth; paginated; Empty → PaginatedAddressResponse)

POST /api/v1/me/addresses — Add address. (+ auth+CSRF; AddressCreateRequest → AddressResponse)

PUT /api/v1/me/addresses/{address_id} — Update address. (+ auth+CSRF; AddressUpdateRequest → AddressResponse)

DELETE /api/v1/me/addresses/{address_id} — Delete address. (+ auth+CSRF; Empty → OKResponse)

GET /api/v1/me/orders — View order history. (+ auth; paginated; Empty → PaginatedOrderSummaryResponse)

GET /api/v1/me/orders/{order_id} — View single order. (+ auth; Empty → OrderDetailResponse)

Catalog & Reviews

GET /api/v1/catalog/categories — List chair categories. (+ public; Empty → CategoryListResponse)

GET /api/v1/catalog/products — Browse/search/sort/filter chairs. (+ public; query params; → PaginatedProductCardResponse)

GET /api/v1/catalog/products/{product_id} — Product details w/ variants, images. (+ public; Empty → ProductDetailResponse)

GET /api/v1/catalog/products/{product_id}/related — Related products. (+ public; paginated; → PaginatedProductCardResponse)

GET /api/v1/catalog/brands — List brands. (+ public; → BrandListResponse)

GET /api/v1/catalog/materials — List materials. (+ public; → MaterialListResponse)

GET /api/v1/catalog/styles — List styles. (+ public; → StyleListResponse)

GET /api/v1/reviews/{product_id} — Reviews list. (+ public; paginated; → PaginatedReviewResponse)

POST /api/v1/reviews/{product_id} — Create review (verified-buyer optional). (+ auth+CSRF; ReviewCreateRequest → ReviewResponse)

PUT /api/v1/reviews/{review_id} — Edit own review. (+ auth+CSRF; ReviewUpdateRequest → ReviewResponse)

DELETE /api/v1/reviews/{review_id} — Delete own review. (+ auth+CSRF; Empty → OKResponse)

Cart & Checkout

GET /api/v1/cart — Get cart (by cookie cart_id or user). (+ public; → CartResponse)

POST /api/v1/cart/items — Add item. (+ public+CSRF; CartItemAddRequest → CartResponse)

PUT /api/v1/cart/items/{item_id} — Update qty. (+ public+CSRF; CartItemUpdateRequest → CartResponse)

DELETE /api/v1/cart/items/{item_id} — Remove item. (+ public+CSRF; Empty → CartResponse)

POST /api/v1/cart/save-for-later — Move item to saved list. (+ public+CSRF; SaveForLaterRequest → CartResponse)

GET /api/v1/cart/saved — Saved-for-later items. (+ public; Empty → SavedItemsResponse)

POST /api/v1/checkout/prepare — Validate cart, compute totals, reserve stock (soft). (+ public+CSRF; CheckoutPrepareRequest → CheckoutPreparedResponse)

POST /api/v1/checkout/orders — Create order (idempotent with key). (+ public+CSRF; OrderCreateRequest → OrderCreatedResponse)

GET /api/v1/checkout/orders/{order_id} — Order status. (+ auth or guest via order token; → OrderDetailResponse)

Payments

POST /api/v1/payments/stripe/intents — Create/confirm Stripe PaymentIntent. (+ public+CSRF; idempotency; StripeIntentRequest → StripeIntentResponse)

POST /api/v1/payments/paypal/orders — Create PayPal order. (+ public+CSRF; PayPalOrderRequest → PayPalOrderResponse)

POST /api/v1/webhooks/stripe — Stripe webhook. (+ public; verified signature; Raw)

POST /api/v1/webhooks/paypal — PayPal webhook. (+ public; verified signature; Raw)

Admin — Orders

GET /api/v1/admin/orders — List orders with filters. (+ role:admin; → PaginatedOrderAdminResponse)

GET /api/v1/admin/orders/{order_id} — Order detail. (+ admin; → OrderAdminDetailResponse)

PUT /api/v1/admin/orders/{order_id}/status — Update status. (+ admin+CSRF; OrderStatusUpdateRequest → OrderAdminDetailResponse)

POST /api/v1/admin/orders/{order_id}/refunds — Record full/partial refund. (+ admin+CSRF; RefundCreateRequest → PaymentRecordResponse)

GET /api/v1/admin/orders/{order_id}/invoice — Generate invoice PDF (URL). (+ admin; → DocumentLinkResponse)

GET /api/v1/admin/orders/{order_id}/shipping-label — Generate label PDF (URL). (+ admin; → DocumentLinkResponse)

POST /api/v1/admin/orders/{order_id}/shipments — Create shipment / tracking. (+ admin+CSRF; ShipmentCreateRequest → ShipmentResponse)

Admin — Products

GET /api/v1/admin/products — List products. (+ admin; → PaginatedProductAdminResponse)

POST /api/v1/admin/products — Create product. (+ admin+CSRF; ProductCreateRequest → ProductDetailResponse)

PUT /api/v1/admin/products/{product_id} — Update product. (+ admin+CSRF; ProductUpdateRequest → ProductDetailResponse)

DELETE /api/v1/admin/products/{product_id} — Delete product. (+ admin+CSRF; Empty → OKResponse)

POST /api/v1/admin/products/{product_id}/variants — Add variant (SKU). (+ admin+CSRF; VariantCreateRequest → VariantResponse)

PUT /api/v1/admin/variants/{variant_id} — Update variant. (+ admin+CSRF; VariantUpdateRequest → VariantResponse)

DELETE /api/v1/admin/variants/{variant_id} — Delete variant. (+ admin+CSRF; Empty → OKResponse)

POST /api/v1/admin/products/{product_id}/images — Upload image. (+ admin+CSRF; multipart; → ImageResponse)

DELETE /api/v1/admin/images/{image_id} — Delete image. (+ admin+CSRF; Empty → OKResponse)

PUT /api/v1/admin/variants/{variant_id}/inventory — Adjust inventory (movement). (+ admin+CSRF; InventoryAdjustRequest → VariantResponse)

PUT /api/v1/admin/products/{product_id}/discount — Set pricing/discount. (+ admin+CSRF; DiscountUpdateRequest → ProductDetailResponse)

Admin — Customers & Support

GET /api/v1/admin/customers — List customers. (+ admin; → PaginatedCustomerResponse)

GET /api/v1/admin/customers/{user_id} — Customer detail + orders. (+ admin; → CustomerDetailResponse)

GET /api/v1/admin/tickets — Support tickets. (+ support/admin; → PaginatedTicketResponse)

POST /api/v1/admin/tickets — Create ticket. (+ support/admin+CSRF; TicketCreateRequest → TicketResponse)

POST /api/v1/admin/tickets/{ticket_id}/messages — Reply. (+ support/admin+CSRF; TicketMessageCreateRequest → TicketMessageResponse)

PUT /api/v1/admin/tickets/{ticket_id}/status — Update status. (+ support/admin+CSRF; TicketStatusUpdateRequest → TicketResponse)

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
    utils.py
  sql/
    schema.sql
    seed.sql
    views.sql
    triggers.sql
  schemas/
    __init__.py
    common.py
    auth.py
    users.py
    addresses.py
    catalog.py
    reviews.py
    cart.py
    checkout.py
    orders.py
    payments.py
    admin.py
    tickets.py
  repositories/
    __init__.py
    user_repository.py
    address_repository.py
    token_repository.py
    login_repository.py
    product_repository.py
    variant_repository.py
    image_repository.py
    category_repository.py
    search_repository.py
    cart_repository.py
    order_repository.py
    payment_repository.py
    shipment_repository.py
    review_repository.py
    ticket_repository.py
  services/
    __init__.py
    auth_service.py
    user_service.py
    address_service.py
    catalog_service.py
    review_service.py
    cart_service.py
    checkout_service.py
    order_service.py
    payment_service.py
    admin_service.py
    ticket_service.py
    email_service.py
    document_service.py
  routers/
    __init__.py
    auth.py
    profile.py
    addresses.py
    catalog.py
    reviews.py
    cart.py
    checkout.py
    payments.py
    webhooks.py
    admin_orders.py
    admin_products.py
    admin_customers.py
    admin_tickets.py

Backend Manifest (JSON)
{
  "project": {
    "name": "chairly-backend",
    "type": "REST API",
    "runtime": "Python FastAPI",
    "database": "SQLite3",
    "data_layer": "raw SQL with aiosqlite"
  },
  "config": {
    "env_vars": [
      { "name": "DATABASE_PATH", "required": true, "example": "./data/app.db", "description": "Absolute/relative path to SQLite database file" },
      { "name": "SECRET_KEY", "required": true, "example": "supersecretlonghex", "description": "JWT signing key (HS256)" },
      { "name": "ACCESS_TOKEN_EXPIRES_MINUTES", "required": true, "example": "15", "description": "Access token TTL minutes" },
      { "name": "REFRESH_TOKEN_EXPIRES_DAYS", "required": true, "example": "7", "description": "Refresh token TTL days (30 when remember-me)" },
      { "name": "CORS_ORIGINS", "required": true, "example": "http://localhost:5173,https://app.example.com", "description": "Comma-separated origin allowlist" },
      { "name": "ENV", "required": true, "example": "development", "description": "Environment name: development|staging|production" },
      { "name": "SMTP_HOST", "required": true, "example": "smtp.sendgrid.net", "description": "SMTP server host" },
      { "name": "SMTP_PORT", "required": true, "example": "587", "description": "SMTP port" },
      { "name": "SMTP_USERNAME", "required": true, "example": "apikey", "description": "SMTP username" },
      { "name": "SMTP_PASSWORD", "required": true, "example": "****", "description": "SMTP password" },
      { "name": "MAIL_FROM", "required": true, "example": "no-reply@chairly.com", "description": "Default sender address" },
      { "name": "STRIPE_SECRET_KEY", "required": false, "example": "sk_test_...", "description": "Stripe API secret for PaymentIntents" },
      { "name": "STRIPE_WEBHOOK_SECRET", "required": false, "example": "whsec_...", "description": "Stripe webhook signing secret" },
      { "name": "PAYPAL_CLIENT_ID", "required": false, "example": "Abc...", "description": "PayPal REST client ID" },
      { "name": "PAYPAL_CLIENT_SECRET", "required": false, "example": "Xyz...", "description": "PayPal REST client secret" },
      { "name": "CSRF_COOKIE_NAME", "required": true, "example": "csrftoken", "description": "Name of CSRF cookie" },
      { "name": "COOKIE_DOMAIN", "required": false, "example": ".example.com", "description": "Cookie domain for prod" }
    ]
  },
  "security": {
    "auth": "JWT in HTTP-only cookies (access + refresh); HS256",
    "refresh_rotation": "Each refresh rotates a new token; old one revoked; reuse detection enforced",
    "csrf": "Double-submit cookie; client sends X-CSRF-Token; validated on unsafe methods",
    "password_hashing": "argon2id (preferred) or bcrypt",
    "cors": "Allowlist from env; credentials true; methods [GET,POST,PUT,DELETE]",
    "rate_limiting": {
      "public": "IP-based: 100 req/10min; auth endpoints 10 req/5min/IP",
      "authenticated": "User-based: 600 req/10min",
      "notes": "Leaky-bucket in memory with per-process limits; adjustable per ENV"
    },
    "rbac": "Role checks in services; roles: customer, admin, support"
  },
  "api": {
    "base_path": "/api/v1",
    "pagination": { "params": ["page", "page_size"], "envelope": "{ items, page, page_size, total }" },
    "errors": { "envelope": "{ error: { code, message, details? } }", "handlers": ["validation", "http", "auth", "db"] },
    "idempotency": "POST non-idempotent; use Idempotency-Key header for create endpoints (orders, payments). PUT is idempotent.",
    "openapi": "Tagged endpoints with examples; schemas via Pydantic v2 models"
  },
  "database": {
    "schema_management": "Execute SQL DDL at startup; CREATE TABLE IF NOT EXISTS; triggers for updated_at",
    "connection_pool": "aiosqlite with connection reuse",
    "schema_file": "app/sql/schema.sql",
    "views_file": "app/sql/views.sql",
    "triggers_file": "app/sql/triggers.sql",
    "uploads_dir": "./data/uploads"
  },
  "tables": [
    {
      "name": "users",
      "description": "User accounts",
      "ddl": "CREATE TABLE IF NOT EXISTS users ( id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, is_email_verified INTEGER NOT NULL DEFAULT 0, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);"]
    },
    {
      "name": "roles",
      "description": "RBAC roles",
      "ddl": "CREATE TABLE IF NOT EXISTS roles ( id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE );",
      "indexes": []
    },
    {
      "name": "user_roles",
      "description": "User-to-role mapping",
      "ddl": "CREATE TABLE IF NOT EXISTS user_roles ( user_id INTEGER NOT NULL, role_id INTEGER NOT NULL, PRIMARY KEY (user_id, role_id), FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id);"]
    },
    {
      "name": "email_verifications",
      "description": "Email verification tokens",
      "ddl": "CREATE TABLE IF NOT EXISTS email_verifications ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, token TEXT NOT NULL UNIQUE, expires_at DATETIME NOT NULL, used_at DATETIME, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_email_verif_user ON email_verifications(user_id);"]
    },
    {
      "name": "password_resets",
      "description": "Password reset tokens",
      "ddl": "CREATE TABLE IF NOT EXISTS password_resets ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, token TEXT NOT NULL UNIQUE, expires_at DATETIME NOT NULL, used_at DATETIME, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": []
    },
    {
      "name": "login_attempts",
      "description": "Track failed and successful logins for lockout",
      "ddl": "CREATE TABLE IF NOT EXISTS login_attempts ( id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, success INTEGER NOT NULL, ip TEXT, user_agent TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_login_attempts_email_time ON login_attempts(email, created_at);"]
    },
    {
      "name": "refresh_tokens",
      "description": "Refresh token rotation + reuse detection",
      "ddl": "CREATE TABLE IF NOT EXISTS refresh_tokens ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, token_id TEXT NOT NULL UNIQUE, parent_token_id TEXT, revoked INTEGER NOT NULL DEFAULT 0, reason TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, last_used_at DATETIME, user_agent TEXT, ip TEXT, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);", "CREATE INDEX IF NOT EXISTS idx_refresh_token_id ON refresh_tokens(token_id);"]
    },
    {
      "name": "addresses",
      "description": "User shipping addresses",
      "ddl": "CREATE TABLE IF NOT EXISTS addresses ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, full_name TEXT NOT NULL, line1 TEXT NOT NULL, line2 TEXT, city TEXT NOT NULL, state TEXT, postal_code TEXT NOT NULL, country TEXT NOT NULL, phone TEXT, is_default INTEGER NOT NULL DEFAULT 0, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_addresses_user ON addresses(user_id);", "CREATE INDEX IF NOT EXISTS idx_addresses_default ON addresses(user_id, is_default);"]
    },
    {
      "name": "brands",
      "description": "Chair brands",
      "ddl": "CREATE TABLE IF NOT EXISTS brands ( id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE );",
      "indexes": []
    },
    {
      "name": "materials",
      "description": "Chair materials",
      "ddl": "CREATE TABLE IF NOT EXISTS materials ( id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE );",
      "indexes": []
    },
    {
      "name": "styles",
      "description": "Chair styles",
      "ddl": "CREATE TABLE IF NOT EXISTS styles ( id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE );",
      "indexes": []
    },
    {
      "name": "categories",
      "description": "Chair categories",
      "ddl": "CREATE TABLE IF NOT EXISTS categories ( id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL UNIQUE, description TEXT );",
      "indexes": []
    },
    {
      "name": "products",
      "description": "Products (chair models)",
      "ddl": "CREATE TABLE IF NOT EXISTS products ( id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL UNIQUE, title TEXT NOT NULL, description TEXT, brand_id INTEGER, style_id INTEGER, active INTEGER NOT NULL DEFAULT 1, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE SET NULL, FOREIGN KEY (style_id) REFERENCES styles(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand_id);", "CREATE INDEX IF NOT EXISTS idx_products_active ON products(active);"]
    },
    {
      "name": "product_categories",
      "description": "Many-to-many products↔categories",
      "ddl": "CREATE TABLE IF NOT EXISTS product_categories ( product_id INTEGER NOT NULL, category_id INTEGER NOT NULL, PRIMARY KEY (product_id, category_id), FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE, FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE );",
      "indexes": []
    },
    {
      "name": "product_images",
      "description": "Images per product",
      "ddl": "CREATE TABLE IF NOT EXISTS product_images ( id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER NOT NULL, url TEXT NOT NULL, alt TEXT, sort_order INTEGER NOT NULL DEFAULT 0, angle TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_product_images_product ON product_images(product_id);"]
    },
    {
      "name": "variants",
      "description": "Product variants (SKU) with price & stock",
      "ddl": "CREATE TABLE IF NOT EXISTS variants ( id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER NOT NULL, sku TEXT NOT NULL UNIQUE, color TEXT, material_id INTEGER, finish TEXT, price_cents INTEGER NOT NULL, currency TEXT NOT NULL DEFAULT 'USD', compare_at_cents INTEGER, stock INTEGER NOT NULL DEFAULT 0, weight_grams INTEGER, dimensions TEXT, active INTEGER NOT NULL DEFAULT 1, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE, FOREIGN KEY (material_id) REFERENCES materials(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_variants_product ON variants(product_id);", "CREATE INDEX IF NOT EXISTS idx_variants_active ON variants(active);"]
    },
    {
      "name": "inventory_movements",
      "description": "Audit of inventory adjustments",
      "ddl": "CREATE TABLE IF NOT EXISTS inventory_movements ( id INTEGER PRIMARY KEY AUTOINCREMENT, variant_id INTEGER NOT NULL, delta INTEGER NOT NULL, reason TEXT NOT NULL, order_id INTEGER, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (variant_id) REFERENCES variants(id) ON DELETE CASCADE, FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_inv_mov_variant ON inventory_movements(variant_id);"]
    },
    {
      "name": "fts_products",
      "description": "FTS5 virtual table for search",
      "ddl": "CREATE VIRTUAL TABLE IF NOT EXISTS fts_products USING fts5(title, description, content='products', content_rowid='id');",
      "indexes": []
    },
    {
      "name": "carts",
      "description": "Shopping carts (user or anonymous via cart_id)",
      "ddl": "CREATE TABLE IF NOT EXISTS carts ( id INTEGER PRIMARY KEY AUTOINCREMENT, cart_key TEXT UNIQUE, user_id INTEGER, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_carts_user ON carts(user_id);", "CREATE INDEX IF NOT EXISTS idx_carts_key ON carts(cart_key);"]
    },
    {
      "name": "cart_items",
      "description": "Items in carts",
      "ddl": "CREATE TABLE IF NOT EXISTS cart_items ( id INTEGER PRIMARY KEY AUTOINCREMENT, cart_id INTEGER NOT NULL, variant_id INTEGER NOT NULL, quantity INTEGER NOT NULL CHECK(quantity > 0), created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(cart_id, variant_id), FOREIGN KEY (cart_id) REFERENCES carts(id) ON DELETE CASCADE, FOREIGN KEY (variant_id) REFERENCES variants(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_cart_items_cart ON cart_items(cart_id);"]
    },
    {
      "name": "saved_items",
      "description": "Save-for-later items",
      "ddl": "CREATE TABLE IF NOT EXISTS saved_items ( id INTEGER PRIMARY KEY AUTOINCREMENT, cart_id INTEGER NOT NULL, variant_id INTEGER NOT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(cart_id, variant_id), FOREIGN KEY (cart_id) REFERENCES carts(id) ON DELETE CASCADE, FOREIGN KEY (variant_id) REFERENCES variants(id) ON DELETE CASCADE );",
      "indexes": []
    },
    {
      "name": "orders",
      "description": "Orders (guest or user)",
      "ddl": "CREATE TABLE IF NOT EXISTS orders ( id INTEGER PRIMARY KEY AUTOINCREMENT, order_number TEXT NOT NULL UNIQUE, user_id INTEGER, guest_email TEXT, status TEXT NOT NULL CHECK(status IN ('pending','processing','paid','shipped','delivered','cancelled','refunded')) DEFAULT 'pending', subtotal_cents INTEGER NOT NULL, tax_cents INTEGER NOT NULL DEFAULT 0, shipping_cents INTEGER NOT NULL DEFAULT 0, discount_cents INTEGER NOT NULL DEFAULT 0, total_cents INTEGER NOT NULL, currency TEXT NOT NULL DEFAULT 'USD', shipping_full_name TEXT NOT NULL, shipping_address TEXT NOT NULL, billing_address TEXT, notes TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);", "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);", "CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);"]
    },
    {
      "name": "order_items",
      "description": "Order line items",
      "ddl": "CREATE TABLE IF NOT EXISTS order_items ( id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, variant_id INTEGER NOT NULL, product_title TEXT NOT NULL, sku TEXT NOT NULL, unit_price_cents INTEGER NOT NULL, quantity INTEGER NOT NULL, total_cents INTEGER NOT NULL, FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE, FOREIGN KEY (variant_id) REFERENCES variants(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);"]
    },
    {
      "name": "payments",
      "description": "Payment records",
      "ddl": "CREATE TABLE IF NOT EXISTS payments ( id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, provider TEXT NOT NULL CHECK(provider IN ('stripe','paypal')), provider_ref TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('requires_payment','authorized','captured','refunded','failed')), amount_cents INTEGER NOT NULL, currency TEXT NOT NULL DEFAULT 'USD', created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_payments_order ON payments(order_id);", "CREATE INDEX IF NOT EXISTS idx_payments_provider_ref ON payments(provider_ref);"]
    },
    {
      "name": "shipments",
      "description": "Shipment tracking",
      "ddl": "CREATE TABLE IF NOT EXISTS shipments ( id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, carrier TEXT NOT NULL, tracking_number TEXT, shipped_at DATETIME, delivered_at DATETIME, label_url TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_shipments_order ON shipments(order_id);"]
    },
    {
      "name": "reviews",
      "description": "Product reviews",
      "ddl": "CREATE TABLE IF NOT EXISTS reviews ( id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER NOT NULL, user_id INTEGER NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5), title TEXT, body TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, verified_purchase INTEGER NOT NULL DEFAULT 0, FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, UNIQUE(product_id, user_id) );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);", "CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id);"]
    },
    {
      "name": "review_votes",
      "description": "Helpful votes",
      "ddl": "CREATE TABLE IF NOT EXISTS review_votes ( id INTEGER PRIMARY KEY AUTOINCREMENT, review_id INTEGER NOT NULL, user_id INTEGER NOT NULL, value INTEGER NOT NULL CHECK(value IN (1,-1)), created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(review_id, user_id), FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE );",
      "indexes": []
    },
    {
      "name": "tickets",
      "description": "Support tickets",
      "ddl": "CREATE TABLE IF NOT EXISTS tickets ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, subject TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('open','pending','closed')) DEFAULT 'open', created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);"]
    },
    {
      "name": "ticket_messages",
      "description": "Messages within tickets",
      "ddl": "CREATE TABLE IF NOT EXISTS ticket_messages ( id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER NOT NULL, author_user_id INTEGER, body TEXT NOT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE, FOREIGN KEY (author_user_id) REFERENCES users(id) ON DELETE SET NULL );",
      "indexes": ["CREATE INDEX IF NOT EXISTS idx_ticket_messages_ticket ON ticket_messages(ticket_id);"]
    }
  ],
  "triggers": [
    "CREATE TRIGGER IF NOT EXISTS trg_users_updated AFTER UPDATE ON users BEGIN UPDATE users SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;",
    "CREATE TRIGGER IF NOT EXISTS trg_variants_updated AFTER UPDATE ON variants BEGIN UPDATE variants SET updated_at=CURRENT_TIMESTAMP WHERE id=NEW.id; END;",
    "CREATE TRIGGER IF NOT EXISTS trg_products_fts_ai AFTER INSERT ON products BEGIN INSERT INTO fts_products(rowid,title,description) VALUES (NEW.id,NEW.title,NEW.description); END;",
    "CREATE TRIGGER IF NOT EXISTS trg_products_fts_au AFTER UPDATE ON products BEGIN INSERT INTO fts_products(fts_products,rowid,title,description) VALUES('delete',OLD.id,OLD.title,OLD.description); INSERT INTO fts_products(rowid,title,description) VALUES (NEW.id,NEW.title,NEW.description); END;",
    "CREATE TRIGGER IF NOT EXISTS trg_products_fts_ad AFTER DELETE ON products BEGIN INSERT INTO fts_products(fts_products,rowid,title,description) VALUES('delete',OLD.id,OLD.title,OLD.description); END;"
  ],
  "entities": [
    { "name": "User", "table": "users", "fields": ["id","email","username","password_hash","is_email_verified","created_at","updated_at"] },
    { "name": "Address", "table": "addresses", "fields": ["id","user_id","full_name","line1","line2","city","state","postal_code","country","phone","is_default"] },
    { "name": "Product", "table": "products", "fields": ["id","slug","title","description","brand_id","style_id","active"] },
    { "name": "Variant", "table": "variants", "fields": ["id","product_id","sku","color","material_id","finish","price_cents","currency","compare_at_cents","stock","weight_grams","dimensions","active"] },
    { "name": "Order", "table": "orders", "fields": ["id","order_number","user_id","guest_email","status","subtotal_cents","tax_cents","shipping_cents","discount_cents","total_cents","currency","shipping_full_name","shipping_address","billing_address","notes"] },
    { "name": "Payment", "table": "payments", "fields": ["id","order_id","provider","provider_ref","status","amount_cents","currency"] },
    { "name": "Shipment", "table": "shipments", "fields": ["id","order_id","carrier","tracking_number","shipped_at","delivered_at","label_url"] },
    { "name": "Review", "table": "reviews", "fields": ["id","product_id","user_id","rating","title","body","verified_purchase"] },
    { "name": "Cart", "table": "carts", "fields": ["id","cart_key","user_id"] },
    { "name": "CartItem", "table": "cart_items", "fields": ["id","cart_id","variant_id","quantity"] }
  ],
  "schemas": {
    "notes": "All Pydantic v2 models with model_config=ConfigDict(...), Field(...), validators; Annotated types for stricter docs.",
    "request": [
      "UserRegisterRequest","LoginRequest","EmailVerifyConfirmRequest","PasswordResetRequest","PasswordResetConfirmRequest","UserUpdateRequest","ChangePasswordRequest",
      "AddressCreateRequest","AddressUpdateRequest",
      "ReviewCreateRequest","ReviewUpdateRequest",
      "CartItemAddRequest","CartItemUpdateRequest","SaveForLaterRequest",
      "CheckoutPrepareRequest","OrderCreateRequest",
      "StripeIntentRequest","PayPalOrderRequest",
      "OrderStatusUpdateRequest","RefundCreateRequest","ShipmentCreateRequest",
      "ProductCreateRequest","ProductUpdateRequest","VariantCreateRequest","VariantUpdateRequest","InventoryAdjustRequest","DiscountUpdateRequest",
      "TicketCreateRequest","TicketMessageCreateRequest","TicketStatusUpdateRequest"
    ],
    "response": [
      "UserResponse","AuthSessionResponse","CsrfTokenResponse","OKResponse","ErrorResponse",
      "AddressResponse","PaginatedAddressResponse",
      "ProductCard","PaginatedProductCardResponse","ProductDetailResponse","BrandListResponse","MaterialListResponse","StyleListResponse","CategoryListResponse",
      "ReviewResponse","PaginatedReviewResponse",
      "CartResponse","SavedItemsResponse","CheckoutPreparedResponse","OrderCreatedResponse","OrderDetailResponse","PaginatedOrderSummaryResponse",
      "StripeIntentResponse","PayPalOrderResponse","PaymentRecordResponse","ShipmentResponse","DocumentLinkResponse",
      "PaginatedOrderAdminResponse","OrderAdminDetailResponse","PaginatedProductAdminResponse","VariantResponse",
      "PaginatedCustomerResponse","CustomerDetailResponse","TicketResponse","TicketMessageResponse","PaginatedTicketResponse"
    ]
  },
  "repositories": [
    { "name": "UserRepository", "file": "app/repositories/user_repository.py",
      "operations": [
        { "method": "create_user", "sql": "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)" },
        { "method": "get_by_email", "sql": "SELECT * FROM users WHERE email = ?" },
        { "method": "get_by_id", "sql": "SELECT * FROM users WHERE id = ?" },
        { "method": "set_email_verified", "sql": "UPDATE users SET is_email_verified=1 WHERE id = ?" },
        { "method": "update_profile", "sql": "UPDATE users SET username=?, updated_at=CURRENT_TIMESTAMP WHERE id=?" }
      ]
    },
    { "name": "TokenRepository", "file": "app/repositories/token_repository.py",
      "operations": [
        { "method": "insert_refresh", "sql": "INSERT INTO refresh_tokens (user_id, token_id, parent_token_id, user_agent, ip) VALUES (?, ?, ?, ?, ?)" },
        { "method": "revoke_token", "sql": "UPDATE refresh_tokens SET revoked=1, reason=? WHERE token_id=?" },
        { "method": "get_token", "sql": "SELECT * FROM refresh_tokens WHERE token_id=?" },
        { "method": "mark_used", "sql": "UPDATE refresh_tokens SET last_used_at=CURRENT_TIMESTAMP WHERE token_id=?" }
      ]
    },
    { "name": "LoginRepository", "file": "app/repositories/login_repository.py",
      "operations": [
        { "method": "record_attempt", "sql": "INSERT INTO login_attempts (email, success, ip, user_agent) VALUES (?, ?, ?, ?)" },
        { "method": "count_recent_failures", "sql": "SELECT COUNT(*) AS c FROM login_attempts WHERE email=? AND success=0 AND created_at >= DATETIME('now', '-15 minutes')" }
      ]
    },
    { "name": "AddressRepository", "file": "app/repositories/address_repository.py",
      "operations": [
        { "method": "list_by_user", "sql": "SELECT * FROM addresses WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?" },
        { "method": "count_by_user", "sql": "SELECT COUNT(*) AS c FROM addresses WHERE user_id=?" },
        { "method": "create", "sql": "INSERT INTO addresses (user_id, full_name, line1, line2, city, state, postal_code, country, phone, is_default) VALUES (?,?,?,?,?,?,?,?,?,?)" },
        { "method": "update", "sql": "UPDATE addresses SET full_name=?, line1=?, line2=?, city=?, state=?, postal_code=?, country=?, phone=?, is_default=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?" },
        { "method": "delete", "sql": "DELETE FROM addresses WHERE id=? AND user_id=?" }
      ]
    },
    { "name": "ProductRepository", "file": "app/repositories/product_repository.py",
      "operations": [
        { "method": "list", "sql": "SELECT * FROM products WHERE active=1 ORDER BY created_at DESC LIMIT ? OFFSET ?" },
        { "method": "count_all", "sql": "SELECT COUNT(*) AS c FROM products WHERE active=1" },
        { "method": "get_detail", "sql": "SELECT * FROM products WHERE id=?" },
        { "method": "search_fts", "sql": "SELECT p.* FROM fts_products f JOIN products p ON p.id=f.rowid WHERE f MATCH ? AND p.active=1 LIMIT ? OFFSET ?" }
      ]
    },
    { "name": "VariantRepository", "file": "app/repositories/variant_repository.py",
      "operations": [
        { "method": "list_by_product", "sql": "SELECT * FROM variants WHERE product_id=? AND active=1" },
        { "method": "get_by_id", "sql": "SELECT * FROM variants WHERE id=?" },
        { "method": "adjust_stock", "sql": "UPDATE variants SET stock=stock+? WHERE id=?" }
      ]
    },
    { "name": "CartRepository", "file": "app/repositories/cart_repository.py",
      "operations": [
        { "method": "get_or_create_by_key", "sql": "SELECT * FROM carts WHERE cart_key=?" },
        { "method": "create_anonymous", "sql": "INSERT INTO carts (cart_key) VALUES (?)" },
        { "method": "link_to_user", "sql": "UPDATE carts SET user_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?" },
        { "method": "get_items", "sql": "SELECT ci.*, v.price_cents, v.currency, v.stock, v.sku FROM cart_items ci JOIN variants v ON v.id=ci.variant_id WHERE ci.cart_id=?" },
        { "method": "upsert_item", "sql": "INSERT INTO cart_items (cart_id, variant_id, quantity) VALUES (?,?,?) ON CONFLICT(cart_id,variant_id) DO UPDATE SET quantity=excluded.quantity, updated_at=CURRENT_TIMESTAMP" },
        { "method": "remove_item", "sql": "DELETE FROM cart_items WHERE id=? AND cart_id=?" },
        { "method": "save_for_later", "sql": "INSERT INTO saved_items (cart_id, variant_id) VALUES (?,?) ON CONFLICT DO NOTHING" },
        { "method": "list_saved", "sql": "SELECT si.*, v.sku, v.price_cents FROM saved_items si JOIN variants v ON v.id=si.variant_id WHERE si.cart_id=?" }
      ]
    },
    { "name": "OrderRepository", "file": "app/repositories/order_repository.py",
      "operations": [
        { "method": "create_order", "sql": "INSERT INTO orders (order_number, user_id, guest_email, status, subtotal_cents, tax_cents, shipping_cents, discount_cents, total_cents, currency, shipping_full_name, shipping_address, billing_address, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)" },
        { "method": "insert_item", "sql": "INSERT INTO order_items (order_id, variant_id, product_title, sku, unit_price_cents, quantity, total_cents) VALUES (?,?,?,?,?,?,?)" },
        { "method": "get_by_id", "sql": "SELECT * FROM orders WHERE id=?" },
        { "method": "list_by_user", "sql": "SELECT * FROM orders WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?" },
        { "method": "count_by_user", "sql": "SELECT COUNT(*) AS c FROM orders WHERE user_id=?" },
        { "method": "update_status", "sql": "UPDATE orders SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?" }
      ]
    },
    { "name": "PaymentRepository", "file": "app/repositories/payment_repository.py",
      "operations": [
        { "method": "record_payment", "sql": "INSERT INTO payments (order_id, provider, provider_ref, status, amount_cents, currency) VALUES (?,?,?,?,?,?)" },
        { "method": "update_status", "sql": "UPDATE payments SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?" },
        { "method": "get_by_provider_ref", "sql": "SELECT * FROM payments WHERE provider=? AND provider_ref=?" }
      ]
    }
  ],
  "services": [
    { "name": "AuthService", "deps": ["UserRepository","TokenRepository","LoginRepository","email_service"], "responsibilities": ["register","login","logout","refresh","verify_email","password_reset","lockout logic","remember-me TTL"] },
    { "name": "UserService", "deps": ["UserRepository"], "responsibilities": ["profile get/update","change password"] },
    { "name": "AddressService", "deps": ["AddressRepository"], "responsibilities": ["CRUD addresses","default handling"] },
    { "name": "CatalogService", "deps": ["ProductRepository","VariantRepository","search_repository"], "responsibilities": ["browse/search/filter/sort","related products"] },
    { "name": "ReviewService", "deps": ["ReviewRepository"], "responsibilities": ["CRUD reviews","verified purchase check"] },
    { "name": "CartService", "deps": ["CartRepository","VariantRepository"], "responsibilities": ["cart lifecycle","save for later","merge carts"] },
    { "name": "CheckoutService", "deps": ["CartRepository","VariantRepository","OrderRepository"], "responsibilities": ["validate cart","compute totals","reserve stock (soft)"] },
    { "name": "OrderService", "deps": ["OrderRepository","InventoryMovements","PaymentRepository"], "responsibilities": ["create orders","status transitions","generate numbers"] },
    { "name": "PaymentService", "deps": ["PaymentRepository","OrderRepository"], "responsibilities": ["Stripe/PayPal intents","webhook handling","refunds"] },
    { "name": "AdminService", "deps": ["OrderRepository","ProductRepository","VariantRepository","UserRepository"], "responsibilities": ["admin CRUD","inventory adjustments","discounts"] },
    { "name": "TicketService", "deps": ["TicketRepository"], "responsibilities": ["manage tickets/messages"] },
    { "name": "EmailService", "deps": [], "responsibilities": ["send transactional emails via SMTP"] },
    { "name": "DocumentService", "deps": [], "responsibilities": ["generate invoice and label documents (URLs)"] }
  ],
  "routers": [
    { "file": "app/routers/auth.py", "tag": "Auth", "endpoints": ["/auth/register","/auth/login","/auth/logout","/auth/refresh","/auth/verify-email/*","/auth/password/*","/auth/csrf"] },
    { "file": "app/routers/profile.py", "tag": "Profile", "endpoints": ["/me","/me/password"] },
    { "file": "app/routers/addresses.py", "tag": "Addresses", "endpoints": ["/me/addresses*"] },
    { "file": "app/routers/catalog.py", "tag": "Catalog", "endpoints": ["/catalog/*"] },
    { "file": "app/routers/reviews.py", "tag": "Reviews", "endpoints": ["/reviews/*"] },
    { "file": "app/routers/cart.py", "tag": "Cart", "endpoints": ["/cart/*"] },
    { "file": "app/routers/checkout.py", "tag": "Checkout", "endpoints": ["/checkout/*"] },
    { "file": "app/routers/payments.py", "tag": "Payments", "endpoints": ["/payments/*"] },
    { "file": "app/routers/webhooks.py", "tag": "Webhooks", "endpoints": ["/webhooks/*"] },
    { "file": "app/routers/admin_orders.py", "tag": "Admin: Orders", "endpoints": ["/admin/orders*"] },
    { "file": "app/routers/admin_products.py", "tag": "Admin: Products", "endpoints": ["/admin/products*","/admin/variants*","/admin/images*"] },
    { "file": "app/routers/admin_customers.py", "tag": "Admin: Customers", "endpoints": ["/admin/customers*"] },
    { "file": "app/routers/admin_tickets.py", "tag": "Admin: Tickets", "endpoints": ["/admin/tickets*"] }
  ],
  "endpoints": [
    {
      "method": "POST",
      "path": "/api/v1/auth/register",
      "summary": "Register new user",
      "description": "Create user and send email verification link.",
      "authentication": "public",
      "path_parameters": [],
      "query_parameters": [],
      "request_body": {
        "schema": "UserRegisterRequest",
        "required": true,
        "example": { "email": "alice@example.com", "username": "alice", "password": "Passw0rd!" }
      },
      "responses": {
        "201": { "description": "Created", "schema": "UserResponse", "example": { "id": 1, "email": "alice@example.com", "username": "alice", "is_email_verified": false } },
        "400": { "description": "Validation error", "schema": "ErrorResponse" },
        "409": { "description": "Conflict (email/username exists)", "schema": "ErrorResponse" }
      },
      "rate_limit": "5/min/IP"
    },
    {
      "method": "GET",
      "path": "/api/v1/catalog/products",
      "summary": "Browse/search products",
      "description": "Supports query, filters and sorting.",
      "authentication": "public",
      "path_parameters": [],
      "query_parameters": [
        { "name": "q", "type": "string", "required": false, "description": "Search text" },
        { "name": "category", "type": "string", "required": false, "description": "Slug of category" },
        { "name": "brand", "type": "string", "required": false },
        { "name": "material", "type": "string", "required": false },
        { "name": "color", "type": "string", "required": false },
        { "name": "price_min", "type": "integer", "required": false, "validation": ">=0 (cents)" },
        { "name": "price_max", "type": "integer", "required": false },
        { "name": "sort", "type": "string", "required": false, "enum": ["price","-price","popularity","-created_at"], "default": "-created_at" },
        { "name": "page", "type": "integer", "default": 1 },
        { "name": "page_size", "type": "integer", "default": 20 }
      ],
      "request_body": null,
      "responses": {
        "200": { "description": "OK", "schema": "PaginatedProductCardResponse" }
      },
      "rate_limit": "200/10min/IP"
    },
    {
      "method": "POST",
      "path": "/api/v1/checkout/orders",
      "summary": "Create order",
      "description": "Validate prepared cart and create order lines; idempotent via Idempotency-Key.",
      "authentication": "public (guest) or authenticated",
      "path_parameters": [],
      "query_parameters": [],
      "request_body": {
        "schema": "OrderCreateRequest",
        "required": true,
        "example": {
          "cart_key": "abc123",
          "shipping": { "full_name": "Alice", "address": { "line1": "123 Main", "city": "NYC", "postal_code": "10001", "country": "US" } },
          "billing_same_as_shipping": true,
          "shipping_method": "standard",
          "email": "alice@example.com"
        }
      },
      "responses": {
        "201": { "description": "Created", "schema": "OrderCreatedResponse", "example": { "order_id": 10, "order_number": "CHAIR-000010", "status": "pending", "total_cents": 25999, "currency": "USD" } },
        "400": { "description": "Invalid cart or out of stock", "schema": "ErrorResponse" }
      },
      "rate_limit": "20/10min/IP"
    }
  ],
  "middleware": [
    "CORS (origins from env, allow credentials)",
    "Request ID (UUID v4 per request) with X-Request-ID propagation",
    "Structured logging (JSON) with path, method, status, duration, req_id",
    "Auth cookie parser; access token verification",
    "Rate limiter (IP/User buckets)",
    "Error handlers returning JSON envelope",
    "CSRF middleware for unsafe methods"
  ],
  "observability": {
    "logging_format": "JSON per line with timestamp, level, logger, message, request_id",
    "request_id_header_in": "X-Request-ID",
    "request_id_header_out": "X-Request-ID"
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

Prefer to read only function definitions of file with the according tool. Only if you need the content of the file read it.
    """
    
    
    llm = ChatOpenAI(
        model="gpt-5-mini", 
        max_retries=15
    )
    
    result = mcp_fast_iterative(llm, "./workspace", message, batch_size=10)
    
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
