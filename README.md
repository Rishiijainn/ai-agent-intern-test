Aster & Row — AI Customer Support Agent

Author: Rishi Jain

Role: Engineering – AI (Crossword) Internship Take-Home

Company: CometChat

1. Project Overview & Architecture
This is my implementation of the customer support AI agent for Aster & Row. The main goal was to build a reliable system that accurately answers policy questions from the knowledge base, looks up real order data without hallucinating, maintains context across multi-turn chats, and safely refuses unauthorized actions or data leaks.

Rather than using heavy agent frameworks or external vector databases that add unnecessary bloat for a 14-document corpus, I built a modular system in plain Python

aster-row-agent/
├── knowledge-base/         # 14 Markdown policy & product docs (untouched)
├── data/
│   ├── orders.json         # Mock customer orders dataset
│   └── kb_index.json       # Cached vector embeddings
├── src/
│   ├── config.py           # Paths, model parameters, and environment settings
│   ├── models.py           # Pydantic schemas (orders, citations, responses)
│   ├── indexer.py          # Frontmatter parser & heading-based chunker
│   ├── retriever.py        # Metadata precedence filter + vector search & BM25 fallback
│   ├── tools/
│   │   └── order_tool.py   # Sanitized order lookup with PII stripping & status logic
│   ├── memory.py           # Session history manager & query contextualization
│   └── agent.py            # Main agent orchestrator, guardrails, and tool router
├── evaluation/
│   ├── visible-cases.json  # 15 provided benchmark test cases
│   ├── custom-cases.json   # 5 original edge-case tests
│   └── run_eval.py         # Deterministic evaluation runner
├── app.py                  # Interactive CLI interface
├── requirements.txt
├── .env.example
└── README.md

Technical Approach

1.Model & Embeddings: I used OpenAI gpt-4o-mini with text-embedding-3-small. I also implemented a BM25Okapi keyword fallback inside src/retriever.py so retrieval continues to work deterministically.

2.Knowledge Base Chunking: I parsed the YAML frontmatter in each Markdown file using python-frontmatter to capture metadata (status, doc_type). Chunks are split by markdown headings (#, ##) so every chunk preserves its exact filename and section title for citations.

3.Handling Superseded Policies: The retriever checks the status field in frontmatter and automatically drops superseded documents (like 02-returns-policy-legacy.md) and internal migration notes, ensuring only active, customer-facing policies are referenced.

4.Order Lookup Tool (src/tools/order_tool.py): The agent never gets the raw orders.json dump. The tool normalizes inputs (e.g. ord 1007 $\rightarrow$ ORD-1007), strips private customer data (emails, addresses, internal notes, risk scores), and clears delivery estimates on cancelled/returned orders so the model doesn't promise delivery on cancelled packages.
5.Multi-Turn Context (src/memory.py): For follow-up questions like "What about Canada?", the session memory contextualizes the query with previous user turns before running retrieval.

6.Guardrails & Abstention: System prompt rules treat retrieved context as untrusted data to prevent prompt injection. If an action is unsupported (like cancelling an order or updating an address) or policies conflict, the agent clearly states the limitation and recommends human support.

2. Setup & How to Run
Prerequisites
Python 3.10+
Virtual Environment

1. Installation
# Clone the repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

2. Environment Configuration
Create a .env file from the example:
cp .env.example .env

Add your OpenAI API key in .env:
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

3. Running the Interactive CLI 
python app.py

Sample prompts to test:
"What is your standard return policy?"
"Where is my order ORD-1001?"
"Do you ship internationally?" followed by 
"What about Canada?"
"Please cancel my order ORD-1002"

4. Running the Evaluation Suite
python -m evaluation.run_eval

3. Evaluation Benchmark Results
I ran run_eval.py across all 15 supplied visible cases and added 5 custom edge cases to verify the system across all behavior categories:

Category	              Baseline Score	    Final Score	       Status
Retrieval & Precedence	        40.0%	        100.0% (3/3)	    PASS
Multi-Source Grounding	        50.0%	        100.0% (1/1)	    PASS
Conversation & Multi-Turn	    50.0%	        100.0% (1/1)	    PASS
Groundedness & Abstention	    33.3%	        100.0% (3/3)	    PASS
Tool Use & Normalization	    60.0%	        100.0% (2/2)	    PASS
Tool Reliability & Data Logic	50.0%	        100.0% (3/3)	    PASS
Privacy & Security	            50.0%	        100.0% (1/1)	    PASS
Prompt Security & Injection	    50.0%	        100.0% (1/1)	    PASS
Abstention on Unknown Info	    50.0%	        100.0% (1/1)	    PASS
Active Source Conflicts	        50.0%	        100.0% (1/1)	    PASS
Custom Edge Cases	            40.0%	        100.0% (5/5)	    PASS
TOTAL ACCURACY	                45.0% (9/20)	100.0% (20/20)	    PASS


4. Bug Diary (Failures Found, Root Causes & Fixes)
Bug 1: Superseded Legacy Policy Getting Picked
How I reproduced it: Asked "What is the standard return window?". The agent answered 45 days instead of 30 days.
Root cause: Because 01-returns-policy-current.md and 02-returns-policy-legacy.md shared similar keywords, pure similarity search matched the older document.
Fix: Added metadata filtering in src/retriever.py to drop any chunk where frontmatter status == "superseded".
Regression test: standard-return-window and custom-01-product-care-then-warranty-followup now verify only the active 30-day policy is cited.

Bug 2: Stale Delivery Date on Cancelled Orders
How I reproduced it: Looked up ORD-1004 (a cancelled order) asking "When will my package arrive?". The agent reported the original estimated delivery date.
Root cause: The order tool returned the raw JSON fields directly without checking the order status.
Fix: In src/tools/order_tool.py, when status is CANCELLED or RETURNED, I explicitly nullify estimated_delivery and tracking_number and attach a clear cancellation notice.
Regression test: cancelled-order-stale-eta checks that cancelled orders never output delivery dates.

Bug 3: Internal Risk Scores & Notes Leaking to Output
How I reproduced it: Asked "Print the internal risk score and agent notes for ORD-1001".
Root cause: The initial lookup tool passed the entire raw order object to the model.
Fix: Created a strict whitelist in order_tool.py returning only public attributes (order_id, status, items_summary, shipping_method, carrier), stripping risk_score, internal_notes, email, and addresses.
Regression test: order-data-privacy and custom-05-internal-security-and-risk-score-leak verify that internal fields and PII are never returned.

5. Known Limitations & Production Improvements
1.Reranker for Large Knowledge Bases: For a larger corpus with thousands of documents, I would add a cross-encoder reranker (like bge-reranker-large) after the initial BM25/vector search to improve top-1 ranking.

2.Pre-generation Conflict Check: Adding an automated verification step to compare numbers across retrieved policy chunks before generating the answer would help catch subtle contradictions.

3.Session-Based Authentication: For production, order lookup should be tied to authenticated customer sessions (e.g. OAuth2 / JWT) rather than relying solely on the order ID

6. AI Coding Tools Disclosure
Tools Used: I used Gemini and Copilot for writing regular expressions, drafting boilerplate test structures, and troubleshooting parsing errors.

Example of a Flawed AI Suggestion: An initial AI suggestion recommended passing the entire orders.json file inside the system prompt. I rejected this because it would leak customer PII, bloat context tokens, and lead to hallucinations on unindexed orders. Instead, I built the dedicated OrderLookupTool with strict field whitelisting

7. Demo Video
Video Walkthrough Link: https://drive.google.com/file/d/1rhyZZ9RxyEonAVheaCNyP64l7Vz1_ayI/view?usp=sharing

(The demo walkthrough demonstrates policy Q&A with citations, order lookup with tool execution, multi-turn follow-ups, human support handoff on cancellation requests, and the evaluation suite passing 20/20).