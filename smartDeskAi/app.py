"""
app.py — Streamlit web UI for SmartDeskAiAssistant (Moxa).

    uv run moxa-ui [-- --reindex]
    uv run streamlit run smartDeskAi/app.py [-- --reindex]
"""

import json
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import utils as ls_utils
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from smartDeskAi.config import (
    DATA_DIR, EMBEDDING_MODEL, LLM_MODEL, PROJECT_ROOT, RERANK_MODEL, RERANK_SCORE_THRESHOLD, SPARSE_MODEL,
)
from smartDeskAi.auth import authenticate
from smartDeskAi.evaluation.answer_eval import DEFAULT_DATASET as ANSWER_DATASET, evaluate_answers
from smartDeskAi.evaluation.answer_eval import load_dataset as load_answer_dataset
from smartDeskAi.evaluation.retrieval_eval import DEFAULT_DATASET as RETRIEVAL_DATASET, evaluate_retrieval
from smartDeskAi.evaluation.retrieval_eval import load_dataset as load_retrieval_dataset
from smartDeskAi.evaluation.routing_eval import DEFAULT_DATASET, evaluate_routing, load_dataset
from smartDeskAi.evaluation.routing_langsmith import langsmith_configured
from smartDeskAi.graph import build_graph
from smartDeskAi.logger import setup_logger
from smartDeskAi.main import TECHNICAL_ERROR, new_session_config
from smartDeskAi.tools.rag import (
    CANDIDATES_PER_RETRIEVER, KB_FILES, TOP_K, get_knowledge_base, kb_entries, rebuild_knowledge_bases,
)

logger = setup_logger("smartDeskAi.app")

GREETING = (
    "Hi, I'm **Moxa**, Synergy Inc's HR & IT assistant. Ask me about HR policies or IT issues, "
    "raise a support ticket, or check the status of one."
)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MOXA_ICON = str(ASSETS_DIR / "moxa.svg")
EMPLOYEE_ICON = str(ASSETS_DIR / "employee.svg")
# Our own script: opens citation links in a slide-out panel instead of a new tab
SOURCE_PANEL = ASSETS_DIR / "source_panel.html"
PRIORITIES = ["Low", "Medium", "High", "Critical"]
# Script argument (after "--") that rebuilds the Qdrant collections from the JSON files on startup
REINDEX_FLAG = "--reindex"
ARCHITECTURE_SVG = PROJECT_ROOT / "docs" / "high_level_architecture.svg"


# ── Shared resources (one per server process) ───────────────

@st.cache_resource(show_spinner="Starting Moxa...")
def get_graph() -> CompiledStateGraph:
    # One compiled graph serves every browser session; sessions are isolated by thread_id
    return build_graph()


@st.cache_resource(show_spinner="Loading knowledge bases...")
def load_knowledge_bases() -> None:
    # Raises on failure so the error is not cached and the next page load retries
    if REINDEX_FLAG in sys.argv[1:]:
        logger.info("Rebuilding the knowledge bases (%s)", REINDEX_FLAG)
        rebuild_knowledge_bases()
    for kb_name in KB_FILES:
        get_knowledge_base(kb_name)


def knowledge_bases_ready() -> bool:
    try:
        load_knowledge_bases()
        return True
    except Exception as e:
        # Keep the chat usable; searches will retry and report the problem politely
        logger.exception("Could not load knowledge bases: %s", e)
        return False


# ── Session state ───────────────────────────────────────────

def start_session() -> None:
    st.session_state.config = new_session_config()
    st.session_state.history = [{"role": "assistant", "content": GREETING}]
    # How many of Moxa's replies in the checkpoint have been copied into history
    st.session_state.replies_shown = 0
    # Answers collected so far for the pending interrupts, keyed by interrupt id
    st.session_state.answers = {}


def pending_interrupts(graph: CompiledStateGraph) -> list[Interrupt]:
    """Interrupts the current thread is waiting on (e.g. a ticket confirmation)."""
    return list(graph.get_state(st.session_state.config).interrupts)


def employee_email(graph: CompiledStateGraph) -> str | None:
    return graph.get_state(st.session_state.config).values.get("employee_email") or signed_in_email()


# ── Sign-in (pre-registered employees) ──────────────────────

def registered_users() -> dict:
    """Employees from the [users] table of .streamlit/secrets.toml (see secrets.toml.example)."""
    try:
        return dict(st.secrets.get("users", {}))
    except FileNotFoundError:
        return {}


def sign_in_defaults() -> tuple[str, str]:
    """Optional pre-filled email and password from the [sign_in_defaults] table of secrets.toml (demos).

    Kept in the git-ignored secrets file so a plaintext password never lands in the code.
    """
    try:
        defaults = st.secrets.get("sign_in_defaults", {})
    except FileNotFoundError:
        return "", ""
    return str(defaults.get("email", "")), str(defaults.get("password", ""))


def auth_configured() -> bool:
    return bool(registered_users())


def signed_in() -> bool:
    # Kept per browser session, so a page refresh signs the employee out
    return "user" in st.session_state


def signed_in_email() -> str | None:
    return st.session_state.user["email"] if signed_in() else None


def sync_replies(graph: CompiledStateGraph) -> None:
    """Copy Moxa's replies from the checkpoint into history.

    Replies are read from the graph rather than from graph.invoke()'s return value, so a reply is never
    lost when Streamlit stops a run early (e.g. the employee clicks again while Moxa is thinking).
    """
    messages = graph.get_state(st.session_state.config).values.get("messages", [])
    replies = [m.text for m in messages if isinstance(m, AIMessage)]
    for reply in replies[st.session_state.replies_shown:]:
        st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.replies_shown = len(replies)


# ── Graph interaction ───────────────────────────────────────

def invoke(graph: CompiledStateGraph, payload) -> None:
    """Run the graph; its reply (if it did not pause for confirmation) is picked up by sync_replies()."""
    try:
        with st.spinner("Moxa is thinking..."):
            graph.invoke(payload, st.session_state.config)
    except Exception as e:
        logger.exception("Turn failed: %s", e)
        st.session_state.history.append({"role": "assistant", "content": TECHNICAL_ERROR})


def show_user_message(text: str) -> None:
    st.session_state.history.append({"role": "user", "content": text})
    with st.chat_message("user", avatar=EMPLOYEE_ICON):
        st.markdown(text)


def send_message(graph: CompiledStateGraph, text: str) -> None:
    show_user_message(text)
    payload = {"messages": [HumanMessage(text)]}
    if email := signed_in_email():
        # Moxa uses the signed-in email for tickets instead of asking for one
        payload |= {"employee_email": email, "email_verified": True}
    invoke(graph, payload)


def resume_if_answered(graph: CompiledStateGraph) -> None:
    """Resume the graph once every pending interrupt has an answer; drop answers to stale interrupts."""
    pending_ids = {item.id for item in pending_interrupts(graph)}
    st.session_state.answers = {k: v for k, v in st.session_state.answers.items() if k in pending_ids}
    if pending_ids and pending_ids <= st.session_state.answers.keys():
        invoke(graph, Command(resume=dict(st.session_state.answers)))
        st.session_state.answers = {}


def answer_interrupt(graph: CompiledStateGraph, interrupt: Interrupt, answer, shown: str) -> None:
    """Record the employee's answer to one interrupt and resume when all of them are answered."""
    st.session_state.answers[interrupt.id] = answer
    show_user_message(shown)
    resume_if_answered(graph)


# ── Rendering ───────────────────────────────────────────────

def render_ticket_confirmation(graph: CompiledStateGraph, interrupt: Interrupt) -> None:
    ticket = interrupt.value["ticket"]
    with st.chat_message("assistant", avatar=MOXA_ICON):
        st.markdown("Here is the ticket I will create for you. Shall I go ahead?")
        with st.container(border=True):
            st.markdown(f"**{ticket['summary']}**")
            st.markdown(ticket["description"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Priority", ticket["priority"])
            col2.metric("Label", ticket["label"])
            col3.markdown(f"**Your email**  \n{ticket['email']}")

        confirm, cancel = st.columns(2)
        if confirm.button("✅ Create ticket", key=f"confirm-{interrupt.id}", type="primary", use_container_width=True):
            answer_interrupt(graph, interrupt, "yes", shown="Yes, create the ticket.")
            st.rerun()
        if cancel.button("❌ Cancel", key=f"cancel-{interrupt.id}", use_container_width=True):
            answer_interrupt(graph, interrupt, "no", shown="No, cancel it.")
            st.rerun()

        with st.expander("Change priority or add details"):
            with st.form(key=f"edit-{interrupt.id}", clear_on_submit=True):
                priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(ticket["priority"])
                                        if ticket["priority"] in PRIORITIES else 1)
                details = st.text_area("Additional details", placeholder="e.g. It fails every morning after login")
                if st.form_submit_button("Update ticket"):
                    # Same wording the CLI accepts, so parse_ticket_edits() handles it
                    # Sent as structured edits so free-text details are never parsed as a priority change
                    edits, shown = {}, []
                    if priority != ticket["priority"]:
                        edits["priority"] = priority
                        shown.append(f"Change the priority to {priority}.")
                    if details.strip():
                        edits["details"] = details.strip()
                        shown.append(f"Add details: {details.strip()}")
                    if edits:
                        answer_interrupt(graph, interrupt, edits, shown=" ".join(shown))
                        st.rerun()
                    st.warning("Change the priority or add some details first.")


def render_interrupt(graph: CompiledStateGraph, interrupt: Interrupt) -> None:
    if isinstance(interrupt.value, dict) and interrupt.value.get("type") == "ticket_confirmation":
        render_ticket_confirmation(graph, interrupt)
    else:
        with st.chat_message("assistant", avatar=MOXA_ICON):
            st.markdown(str(interrupt.value))


def render_sidebar(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    with st.sidebar:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.image(MOXA_ICON, width=48, alt="")
            st.markdown("# Moxa<sup style='font-size: 0.5em'>©</sup>", unsafe_allow_html=True, width="content")
        st.caption("Synergy Inc · HR & IT Smart Assistant")
        if st.button("➕ New session", use_container_width=True):
            start_session()
            st.rerun()

        st.divider()
        email = employee_email(graph)
        st.markdown(f"**Your email:** {email}" if email else "**Your email:** _not provided yet_")
        if not kb_loaded:
            st.warning("The knowledge bases could not be loaded right now; answers may be limited.")

        st.divider()
        st.markdown(
            "**Try asking**\n"
            "- What's the dress code?\n"
            "- How do I reset my password?\n"
            "- Raise a ticket: my laptop screen is cracked\n"
            "- What's the status of my tickets?"
        )

        st.divider()
        st.caption("© 2026 Synergy Inc. The Moxa name is copyrighted. All rights reserved.")


@st.cache_data
def readme_diagram(heading: str) -> str | None:
    """Return the first Mermaid block under the given README.md heading line, if any."""
    try:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    except OSError:
        logger.warning("README.md not found; skipping %r diagram", heading)
        return None
    match = re.search(rf"^{re.escape(heading)}\s.*?^```mermaid\n(.*?)^```", readme, re.S | re.M)
    return match.group(1) if match else None


def show_high_level_architecture(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    st.markdown(
        "Moxa is a multi-agent HR & IT chatbot for employees at Synergy Inc. that uses the company's HR and IT "
        "knowledge bases to answer employee questions, helps them create Jira tickets, and checks the status "
        "of their tickets.\n\n"
        "Architecture diagram below describes internals. It has both UI and Text modes. UI is the default mode "
        "and implemented by Streamlit framework. It has agents : Ochestrator, HR and IT, and Synthesizer, uses Hybrid RAG "
        "for both HR and IT knowledge bases to ground answers, uses Jira tools for ticketing.\n\n"
        "The arrows show how a question flows through Moxa: from the web UI or CLI, through the orchestrator to the HR and/or IT agent" 
        "and their tools, then the synthesizer merges the answers into one cited reply."
    )
    if ARCHITECTURE_SVG.exists():
        # The SVG has only a viewBox, no fixed size, so cap its width and center it in the dialog
        with st.container(horizontal_alignment="center"):
            st.image(str(ARCHITECTURE_SVG), width=560)
    else:
        logger.warning("%s not found; skipping architecture image", ARCHITECTURE_SVG)
        st.info("The architecture diagram (docs/high_level_architecture.svg) is missing.")


def show_architecture(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    st.markdown(
        "Moxa is a LangGraph `StateGraph` of four agents:\n"
        "- `Orchestrator` routes each question to the HR and/or "
        "IT specialist (both run in parallel when a question spans the two)\n"
        "- `hr_agent` / `it_agent` search their knowledge base and create, list, or check Jira tickets\n"
        "- `synthesizer` merges the specialists' answers into one reply\n\n"
        "Ticket creation pauses the graph with `interrupt()` until the employee confirms. "
        "A checkpointer keeps each session's memory, keyed by its thread ID.\n\n"
        f"**LLM:** `{LLM_MODEL}`"
    )
    diagram = readme_diagram("## Architecture")
    if diagram:
        st.mermaid_chart(diagram, alt="Moxa multi-agent architecture")


def show_hybrid_rag(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    st.markdown(
        "Each knowledge-base search runs two retrievers over the same Qdrant collection:\n"
        f"- **Dense:** OpenAI embeddings (`{EMBEDDING_MODEL}`), cosine similarity\n"
        f"- **Sparse:** BM25 term weights (`{SPARSE_MODEL}`)\n\n"
        f"The top {CANDIDATES_PER_RETRIEVER} results from each are fused with Reciprocal Rank Fusion, "
        f"then a cross-encoder (`{RERANK_MODEL}`) re-ranks them and the best {TOP_K} are kept. "
        f"Chunks scoring below `{RERANK_SCORE_THRESHOLD}` are not treated as confident matches."
    )
    diagram = readme_diagram("### Retrieval pipeline (one per knowledge base)")
    if diagram:
        st.markdown("**Retrieval pipeline (one per knowledge base)**")
        st.mermaid_chart(diagram, alt="Hybrid RAG retrieval pipeline")

    # The dialog is a fragment, so these clicks rerun only the dialog and keep it open
    with st.container(horizontal=True):
        for kb_name in KB_FILES:
            if st.button(f"View {kb_name.upper()} KB", icon=":material/table_view:"):
                st.session_state.rag_kb_view = kb_name
        with st.popover("Rebuild RAG", icon=":material/refresh:"):
            st.warning(
                "This restarts Moxa for everyone and re-indexes both knowledge bases from their JSON files. "
                "Open chats and sign-ins are lost, and Moxa is unavailable for a minute or so.",
                icon=":material/warning:",
            )
            if st.button("Restart and rebuild", type="primary", icon=":material/restart_alt:"):
                restart_with_reindex()
                st.info("Moxa is restarting. This page reconnects by itself; sign in again once it's back.")
    kb_name = st.session_state.get("rag_kb_view")
    if kb_name:
        entries = kb_json(kb_name)
        st.markdown(f"**{kb_name.upper()} knowledge base** · `{KB_FILES[kb_name]}` · {len(entries)} entries")
        st.dataframe(
            entries,
            hide_index=True,
            column_order=["id", "category", "subcategory", "question", "answer"],
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "category": "Category",
                "subcategory": "Subcategory",
                "question": st.column_config.TextColumn("Question", width="medium"),
                "answer": st.column_config.TextColumn("Answer", width="large"),
            },
        )


def restart_with_reindex() -> None:
    """Replace this Streamlit server with a fresh one started with --reindex.

    Streamlit drops its own flags from sys.argv, so the ones it was launched with are rebuilt from its
    config. The swap waits a moment so this run can tell the browser first.
    """
    from streamlit import config as st_config

    flags = {
        key: value for key in st_config._config_options
        if st_config.get_where_defined(key) == st_config._DEFINED_BY_FLAG
        and isinstance(value := st_config.get_option(key), (str, int, float, bool))
    }
    # Streamlit re-records the port once bound, so it never looks flag-defined; pass the live one
    flags["server.port"] = st_config.get_option("server.port")
    # Don't open another browser tab on restart; the current one reconnects
    flags["server.headless"] = True
    script_args = [arg for arg in sys.argv[1:] if arg != REINDEX_FLAG]
    argv = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        *(f"--{key}={value}" for key, value in flags.items()), "--", *script_args, REINDEX_FLAG,
    ]
    logger.warning("Restarting Moxa to rebuild the knowledge bases: %s", " ".join(argv))

    def restart() -> None:
        logging.shutdown()  # flush the log before the process is replaced
        # Some descriptors (e.g. the listening socket) survive exec, and an inherited, unserved socket
        # hangs browsers. Mark them close-on-exec rather than closing them: closing them under the
        # server's running threads crashes the process before exec happens.
        for fd in map(int, os.listdir("/dev/fd")):
            if fd > 2:
                try:
                    os.set_inheritable(fd, False)
                except OSError:
                    pass  # already closed, or the /dev/fd listing's own descriptor
        os.execv(sys.executable, argv)

    threading.Timer(1.5, restart).start()


@st.cache_data
def kb_json(kb_name: str) -> list[dict]:
    """The raw Q&A entries of a knowledge-base JSON file."""
    with open(DATA_DIR / KB_FILES[kb_name], encoding="utf-8") as f:
        return json.load(f)


def show_session_info(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    state = graph.get_state(st.session_state.config)
    st.markdown(
        f"- **Thread ID:** `{st.session_state.config['configurable']['thread_id']}`\n"
        f"- **Employee email:** {state.values.get('employee_email') or '_not provided yet_'}\n"
        f"- **Messages in memory:** {len(state.values.get('messages', []))}\n"
        f"- **Pending confirmations:** {len(state.interrupts)}"
    )

    st.markdown("**Last routing decision**")
    agents = state.values.get("selected_agents")
    latest = next((m for m in reversed(state.values.get("messages", [])) if isinstance(m, HumanMessage)), None)
    if not agents or latest is None:
        st.caption("No question has been routed yet. Ask Moxa something first.")
        return
    st.mermaid_chart(routing_diagram(latest.text, agents), alt="Last routing decision")
    if reasoning := state.values.get("routing_reasoning"):
        st.caption(f"**Orchestrator's reasoning:** {reasoning}")


def mermaid_label(text: str, limit: int = 80) -> str:
    """Employee text made safe for a quoted Mermaid label: one line, shortened, no raw quotes or tags."""
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;").replace("`", "'")


def routing_diagram(query: str, agents: list[str]) -> str:
    """A flowchart of one turn: routed agents highlighted, the skipped one greyed out."""
    lines = [
        "flowchart LR",
        f'    Q(["{mermaid_label(query)}"]) --> O["orchestrator"]',
    ]
    for node, agent in (("HR", "hr_agent"), ("IT", "it_agent")):
        routed = agent in agents
        lines.append(f'    O {'==>|"Send()"|' if routed else '-.->|"not routed"|'} {node}["{agent}"]')
        lines.append(f'    {node} {"==>" if routed else "-.->"} S')
        lines.append(f"    class {node} {'routed' if routed else 'skipped'}")
    lines += [
        '    S["synthesizer"] --> R(["reply"])',
        "    classDef routed fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b",
        "    classDef skipped fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray:4 3,color:#9ca3af",
    ]
    return "\n".join(lines)


def show_monitoring(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    """LangSmith tracing status for this deployment and session."""
    if not ls_utils.tracing_is_enabled():
        st.info(
            "Add these to `.env` and restart Moxa to trace every agent, tool, and LLM call:\n\n"
            "```\nLANGSMITH_TRACING=true\nLANGSMITH_API_KEY=your-langsmith-key-here\n"
            "LANGSMITH_PROJECT=Moxa SmartDesk AI\n```",
            title="LangSmith tracing is off",
            icon=":material/visibility_off:",
        )
        return
    st.success("Every agent, tool, and LLM call is being traced.", title="LangSmith tracing is on")
    st.markdown(
        f"- **Project:** `{ls_utils.get_tracer_project()}`\n"
        f"- **Endpoint:** `{os.environ.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}`\n"
        f"- **This session's traces:** filter by metadata `thread_id = "
        f"{st.session_state.config['configurable']['thread_id']}`"
    )
    st.link_button("Open LangSmith", "https://smith.langchain.com", icon=":material/open_in_new:")


# (layer, technology, what Moxa uses it for)
TECH_STACK = [
    ("Language", "Python", "Everything: agents, tools, web UI and CLI"),
    ("Agent orchestration", "LangGraph",
     "StateGraph of the orchestrator, HR/IT agents and synthesizer; interrupt() for ticket confirmation; "
     "in-memory checkpointer for session memory"),
    ("LLM framework", "LangChain", "Tool-calling agents and the @tool ticket and search tools"),
    ("LLM", "OpenAI", f"{LLM_MODEL} for routing, answering and synthesis"),
    ("Embeddings", "OpenAI", f"{EMBEDDING_MODEL} dense vectors for the knowledge bases"),
    ("Vector store", "Qdrant", "Dense + sparse hybrid search over the HR and IT knowledge bases"),
    ("Sparse & re-ranking", "FastEmbed", f"BM25 ({SPARSE_MODEL}) and cross-encoder re-ranker ({RERANK_MODEL})"),
    ("Ticketing", "Jira Cloud", "Create, list, search and check support tickets"),
    ("Web UI", "Streamlit", "This chat app, sign-in and the About Moxa views"),
    ("Observability", "LangSmith", "Traces of every agent, tool and LLM call"),
    ("Validation", "Pydantic", "Structured LLM output such as the orchestrator's routing decision"),
    ("Authentication", "scrypt (hashlib)", "Hashed employee passwords in Streamlit secrets"),
    ("Packaging", "uv + Hatchling", "Dependency locking and the moxa / moxa-ui commands"),
]


def show_tech_stack(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    st.markdown("The libraries and services Moxa is built on.")
    rows = [{"Layer": layer, "Technology": tech, "Used for": purpose} for layer, tech, purpose in TECH_STACK]
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"Used for": st.column_config.TextColumn("Used for", width="large")})


# Planned improvements, shown as a bullet list
ROADMAP = [
    "RBAC control: the About Moxa menu is shown only to IT administrators",
    "Persistent memory for chats per user",
    "Semantic caching",
]


def show_roadmap(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    st.markdown("What's planned for Moxa next.")
    st.markdown("\n".join(f"- {item}" for item in ROADMAP))


EVAL_COMMAND = "uv run moxa-eval-routing --langsmith"
RETRIEVAL_EVAL_COMMAND = "uv run moxa-eval-retrieval --langsmith"
ANSWER_EVAL_COMMAND = "uv run moxa-eval-answers --langsmith"


def show_evaluate(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    """Routing, retrieval and answer evaluations, each run as a LangSmith experiment."""
    if not langsmith_configured():
        st.warning("Add `LANGSMITH_API_KEY` to `.env` and restart Moxa to run evaluations.", icon=":material/key_off:")
    routing, answers, retrieval = st.tabs(["Routing", "Faithfulness & Correctness", "Retrieval"])
    with routing:
        show_routing_eval()
    with answers:
        show_answer_eval(kb_loaded)
    with retrieval:
        show_retrieval_eval(kb_loaded)


def show_routing_eval() -> None:
    """Run the routing evaluation as a LangSmith experiment, same as EVAL_COMMAND."""
    cases = len(load_dataset(DEFAULT_DATASET))
    st.markdown(
        f"Checks that the orchestrator sends each question to the right agent(s). The {cases} labeled "
        "conversations in `routing_dataset.json` (HR, IT, both, small talk, out of scope, ticket status, "
        "follow-ups and topic switches) are routed by the live router and scored as a LangSmith experiment. "
        "Same as running:"
    )
    st.code(EVAL_COMMAND, language="bash")
    # The dialog is a fragment, so this click reruns only the dialog and keeps it open
    if st.button("Run routing evaluation", type="primary", icon=":material/play_arrow:",
                 disabled=not langsmith_configured()):
        with st.spinner(f"Routing {cases} conversations and scoring them in LangSmith..."):
            try:
                st.session_state.routing_eval = evaluate_routing(langsmith=True)
            except Exception:
                logger.exception("Routing evaluation failed")
                st.error("The evaluation could not run. Details are in `log/app.log`.", icon=":material/error:")

    report = st.session_state.get("routing_eval")
    if not report:
        return
    summary = report.summary
    failures = [r for r in report.results if not r.correct]
    st.divider()
    with st.container(horizontal=True):
        st.metric("Routing accuracy", f"{summary['accuracy']:.1%}", border=True)
        st.metric("Keyword fallback", f"{summary['keyword_fallback']['accuracy']:.1%}", border=True,
                  help="Accuracy of the keyword router Moxa falls back to when the LLM is unavailable")
        st.metric("Misrouted", f"{len(failures)} / {summary['cases']}", border=True)
        st.metric("Latency (p50)", f"{summary['latency_s']['p50']:.2f}s", border=True)
    if report.experiment_url:
        st.link_button("Open experiment in LangSmith", report.experiment_url, icon=":material/open_in_new:")

    left, right = st.columns(2)
    with left:
        st.markdown("**Per agent**")
        st.dataframe(
            [{"Agent": agent, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "Cases": m["support"]}
             for agent, m in summary["per_agent"].items()],
            hide_index=True,
            column_config={name: st.column_config.NumberColumn(format="percent") for name in ("Precision", "Recall", "F1")},
        )
    with right:
        st.markdown("**Per category**")
        st.dataframe(
            [{"Category": category.replace("_", " "), "Accuracy": m["accuracy"], "Cases": m["cases"]}
             for category, m in summary["per_category"].items()],
            hide_index=True,
            column_config={"Accuracy": st.column_config.NumberColumn(format="percent")},
        )

    st.markdown("**Misrouted cases**")
    if failures:
        st.dataframe(
            [{"Case": r.id, "Latest message": r.latest_message, "Expected": ", ".join(r.expected),
              "Got": r.predicted if isinstance(r.predicted, str) else ", ".join(r.predicted),
              "Router's reasoning": r.reasoning} for r in failures],
            hide_index=True,
        )
    else:
        st.success("Every conversation was routed to the expected agent(s).", icon=":material/check_circle:")
    st.caption(f"Full results: `{report.output.relative_to(PROJECT_ROOT)}`")


def show_retrieval_eval(kb_loaded: bool) -> None:
    """Run the retrieval evaluation as a LangSmith experiment, same as RETRIEVAL_EVAL_COMMAND."""
    cases = load_retrieval_dataset(RETRIEVAL_DATASET)
    out_of_scope = sum(not case["expected"] for case in cases)
    st.markdown(
        f"Checks that hybrid search finds the right knowledge-base entry. The {len(cases)} labeled questions in "
        f"`retrieval_dataset.json` (paraphrases, short keywords, typos, and {out_of_scope} out-of-scope questions "
        "that should find nothing) are searched in the live Qdrant collections. Each stage (dense, BM25, "
        "RRF fusion, re-ranking) is scored, then the full pipeline with its confidence threshold. Same as running:"
    )
    st.code(RETRIEVAL_EVAL_COMMAND, language="bash")
    if not kb_loaded:
        st.warning("The knowledge base isn't loaded, so retrieval can't be evaluated.", icon=":material/warning:")
    if st.button("Run retrieval evaluation", type="primary", icon=":material/play_arrow:",
                 disabled=not (langsmith_configured() and kb_loaded)):
        with st.spinner(f"Searching {len(cases)} questions and scoring them in LangSmith..."):
            try:
                st.session_state.retrieval_eval = evaluate_retrieval(langsmith=True)
            except Exception:
                logger.exception("Retrieval evaluation failed")
                st.error("The evaluation could not run. Details are in `log/app.log`.", icon=":material/error:")

    report = st.session_state.get("retrieval_eval")
    if not report:
        return
    summary = report.summary
    k = summary["top_k"]
    failures = [r for r in report.results if not r.correct]
    st.divider()
    with st.container(horizontal=True):
        st.metric("Pipeline accuracy", f"{summary['accuracy']:.1%}", border=True,
                  help="Right entry returned, or nothing returned for an out-of-scope question")
        st.metric(f"Re-ranked Hit@{k}", f"{summary['stages']['reranked'][f'hit@{k}']:.1%}", border=True,
                  help=f"An expected entry is in the top {k} after re-ranking, before the confidence threshold")
        st.metric("False rejections", f"{summary['false_rejections']:.1%}", border=True,
                  help="Answerable questions where no entry passed the confidence threshold")
        st.metric("Correct rejections", f"{summary['correct_rejections']:.1%}", border=True,
                  help="Out-of-scope questions where nothing was returned")
    if report.experiment_url:
        st.link_button("Open experiment in LangSmith", report.experiment_url, icon=":material/open_in_new:")

    percent = st.column_config.NumberColumn(format="percent")
    st.markdown("**Per stage** (answerable questions)")
    st.dataframe(
        [{"Stage": stage, "Hit@1": m["hit@1"], f"Hit@{k}": m[f"hit@{k}"], f"Recall@{k}": m[f"recall@{k}"],
          "MRR": round(m["mrr"], 3)} for stage, m in summary["stages"].items()],
        hide_index=True,
        column_config={"Hit@1": percent, f"Hit@{k}": percent, f"Recall@{k}": percent},
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Per category**")
        st.dataframe(
            [{"Category": category.replace("_", " "), "Accuracy": m["accuracy"], "Cases": m["cases"]}
             for category, m in summary["per_category"].items()],
            hide_index=True, column_config={"Accuracy": percent},
        )
    with right:
        st.markdown(f"**Threshold sweep** (current {summary['threshold']})")
        st.dataframe(
            [{"Threshold": row["threshold"], "Answered": row["answered_correctly"],
              "False rej.": row["false_rejections"], "Correct rej.": row["correct_rejections"]}
             for row in summary["threshold_sweep"]],
            hide_index=True,
            column_config={name: percent for name in ("Answered", "False rej.", "Correct rej.")},
        )

    st.markdown("**Failed questions**")
    if failures:
        st.dataframe(
            [{"Case": r.id, "Question": r.query, "Expected": ", ".join(r.expected) or "nothing",
              "Returned": ", ".join(r.returned) or "nothing",
              f"Top {k} (re-rank score)": r.error or ", ".join(
                  f"{kb_id} ({score:.2f})" for kb_id, score in zip(r.rankings["reranked"][:k], r.scores[:k]))}
             for r in failures],
            hide_index=True,
        )
    else:
        st.success("Every question found the expected entry, or nothing when out of scope.", icon=":material/check_circle:")
    st.caption(f"Full results: `{report.output.relative_to(PROJECT_ROOT)}`")


def show_answer_eval(kb_loaded: bool) -> None:
    """Run the answer evaluation as a LangSmith experiment, same as ANSWER_EVAL_COMMAND."""
    cases = load_answer_dataset(ANSWER_DATASET)
    out_of_scope = sum(not case["expected_sources"] for case in cases)
    st.markdown(
        f"Checks that Moxa's final answers are faithful and correct. The {len(cases)} questions in "
        f"`answer_dataset.json` (HR, IT, mixed HR + IT, and {out_of_scope} out-of-scope questions) run through "
        "the full graph. An LLM judge then checks each answer's claims against the knowledge-base results "
        "the agents retrieved (faithfulness), and compares the answer with a reference (correctness). Same as running:"
    )
    st.code(ANSWER_EVAL_COMMAND, language="bash")
    if not kb_loaded:
        st.warning("The knowledge base isn't loaded, so answers can't be evaluated.", icon=":material/warning:")
    if st.button("Run answer evaluation", type="primary", icon=":material/play_arrow:",
                 disabled=not (langsmith_configured() and kb_loaded)):
        with st.spinner(f"Answering {len(cases)} questions and grading them in LangSmith..."):
            try:
                st.session_state.answer_eval = evaluate_answers(langsmith=True)
            except Exception:
                logger.exception("Answer evaluation failed")
                st.error("The evaluation could not run. Details are in `log/app.log`.", icon=":material/error:")

    report = st.session_state.get("answer_eval")
    if not report:
        return
    summary = report.summary
    st.divider()
    with st.container(horizontal=True):
        st.metric("Faithfulness", f"{summary['faithfulness']:.1%}", border=True,
                  help="Mean share of each answer's claims supported by the retrieved knowledge-base entries")
        st.metric("Correctness", f"{summary['correctness']:.1%}", border=True,
                  help="Against the reference answer: correct 1, partially correct 0.5, incorrect 0")
        st.metric("Source hit", f"{summary['source_hit']:.1%}", border=True,
                  help="Answerable questions whose answer cites an expected knowledge-base entry")
        st.metric("Correct abstentions", f"{summary['correct_abstentions']:.1%}", border=True,
                  help="Out-of-scope questions answered without citing anything")
    if report.experiment_url:
        st.link_button("Open experiment in LangSmith", report.experiment_url, icon=":material/open_in_new:")

    percent = st.column_config.NumberColumn(format="percent")
    st.markdown("**Per category**")
    st.dataframe(
        [{"Category": category.replace("_", " "), "Faithfulness": m["faithfulness"], "Correctness": m["correctness"],
          "Cases": m["cases"]} for category, m in summary["per_category"].items()],
        hide_index=True, column_config={"Faithfulness": percent, "Correctness": percent},
    )

    flagged = [r for r in report.results if r.error or r.faithfulness < 1.0 or r.correctness < 1.0]
    st.markdown("**Flagged answers**")
    if flagged:
        st.dataframe(
            [{"Case": r.id, "Question": r.question, "Faithfulness": r.faithfulness, "Verdict": r.correctness_verdict.replace("_", " "),
              "Unsupported claims": "; ".join(r.unsupported_claims),
              "Issues": r.error or "; ".join([*(f"wrong: {f}" for f in r.wrong_facts), *(f"missing: {f}" for f in r.missing_facts)]),
              "Answer": r.answer} for r in flagged],
            hide_index=True, column_config={"Faithfulness": percent},
        )
    else:
        st.success("Every answer was fully grounded in the knowledge base and matched its reference.",
                   icon=":material/check_circle:")
    st.caption(f"Full results: `{report.output.relative_to(PROJECT_ROOT)}`")


INTERNALS = {
    "Architecture": (":material/schema:", show_high_level_architecture),
    "High Level Design": (":material/account_tree:", show_architecture),
    "Technical Stack": (":material/stacks:", show_tech_stack),
    "Hybrid RAG": (":material/manage_search:", show_hybrid_rag),
    "Session Info": (":material/forum:", show_session_info),
    "Monitor": (":material/monitoring:", show_monitoring),
    "Evaluate": (":material/fact_check:", show_evaluate),
    "Roadmap": (":material/route:", show_roadmap),
}


# ── Ratings ─────────────────────────────────────────────────

RATINGS_DIR = PROJECT_ROOT / "rating"
RATINGS_FILE = RATINGS_DIR / "ratings.jsonl"


def save_rating(stars: int, comment: str) -> None:
    rating = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stars": stars,
        "comment": comment.strip(),
        "thread_id": st.session_state.config["configurable"]["thread_id"],
        "email": employee_email(get_graph()),
    }
    RATINGS_DIR.mkdir(exist_ok=True)
    with open(RATINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rating) + "\n")
    logger.info("Moxa rated %d/5", stars)


@st.dialog("Rate Moxa", icon=":material/star:")
def rate_moxa() -> None:
    with st.form("rate_moxa", border=False):
        st.markdown("How helpful was Moxa today?")
        selected = st.feedback("stars")
        comment = st.text_area("Anything Moxa could do better? (optional)")
        if st.form_submit_button("Submit", type="primary"):
            if selected is None:
                st.warning("Please pick a star rating.")
            else:
                save_rating(selected + 1, comment)  # st.feedback returns 0-4
                st.session_state.rated = True
                st.rerun()


def render_landing() -> None:
    """The sign-in page every employee sees first; Moxa itself loads only after sign-in."""
    with st.container(horizontal_alignment="center"):
        st.markdown("# HR and IT Portal", text_alignment="center")
        st.markdown("### Synergy Inc", text_alignment="center")
        with st.container(width=420):
            if not auth_configured():
                st.error(
                    "Sign-in isn't set up yet. Add employees to `.streamlit/secrets.toml` "
                    "(see `secrets.toml.example`) and restart the portal.",
                    icon=":material/lock:",
                )
                return
            default_email, default_password = sign_in_defaults()
            with st.form("sign_in"):
                email = st.text_input("Work email", value=default_email)
                password = st.text_input("Password", value=default_password, type="password")
                if st.form_submit_button("Sign in", type="primary", icon=":material/login:", width="stretch"):
                    user = authenticate(registered_users(), email, password)
                    if user is None:
                        logger.info("Failed sign-in for %r", email.strip().lower())
                        st.error("Incorrect email or password.")
                    else:
                        st.session_state.user = user
                        logger.info("Signed in %s", user["email"])
                        st.rerun()
            st.caption("Sign in to chat with Moxa, Synergy Inc's HR & IT assistant.", text_alignment="center")


def render_account() -> None:
    """The signed-in employee's name, with a Sign out option."""
    if st.menu_button(st.session_state.user["name"], ["Sign out"], icon=":material/account_circle:",
                      type="tertiary", format_func=lambda option: f":material/logout: {option}"):
        del st.session_state.user
        # The signed-in email is locked into the conversation, so sign-out starts a fresh one
        start_session()
        st.rerun()


def render_toolbar(graph: CompiledStateGraph, kb_loaded: bool) -> None:
    """Rate Moxa, the 'About Moxa' menu and the employee's account, pinned into the top-right toolbar in place of Deploy."""
    # Streamlit has no API for toolbar items, so a keyed container is moved there with CSS
    st.html(
        """
        <style>
        .st-key-moxa_toolbar {
            position: fixed;
            top: 0.75rem;
            right: 3.5rem;
            z-index: 999991;
            width: auto;
        }
        /* Toolbar labels at 16px instead of Streamlit's 14px */
        .st-key-moxa_toolbar button p {
            font-size: 1rem;
        }
        </style>
        """
    )
    with st.container(key="moxa_toolbar", horizontal=True, vertical_alignment="center"):
        if st.button("Rate Moxa", icon=":material/star:", type="tertiary"):
            rate_moxa()
        choice = st.menu_button(
            "About Moxa", list(INTERNALS), icon=":material/account_tree:", type="tertiary",
            format_func=lambda name: f"{INTERNALS[name][0]} {name}",
        )
        render_account()
    if st.session_state.pop("rated", False):
        st.toast("Thanks for rating Moxa!", icon=":material/favorite:")
    if choice:
        icon, show = INTERNALS[choice]
        st.dialog(choice, icon=icon, width="large")(show)(graph, kb_loaded)


# ── Citations ───────────────────────────────────────────────

# A Sources entry from add_citations(): "- [1] HR knowledge base · Category: "Question?" (hr_kb0001)"
SOURCE_LINE = re.compile(r"^- \[(\d+)\] (.+) \(((?:hr|it)_kb\d+)\)$", re.MULTILINE)


SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def link_citations(text: str) -> str:
    """Show a reply's [n] citations as superscript ⁽ⁿ⁾ and make them, and the Sources entries,
    open the knowledge-base entry (in the slide-out panel; a new tab without JavaScript). Unicode superscripts avoid rendering the reply as HTML."""
    body, sep, sources = text.rpartition("\n\nSources:\n")
    ids = {n: kb_id for n, _, kb_id in SOURCE_LINE.findall(sources)}
    if not sep or not ids:
        return text
    body = re.sub(
        r"\s*\[(\d+)\]",
        lambda m: f"[⁽{m[1].translate(SUPERSCRIPT)}⁾](?source={ids[m[1]]})" if m[1] in ids else m[0],
        body,
    )
    sources = SOURCE_LINE.sub(lambda m: f"- [[{m[1]}] {m[2]}](?source={m[3]})", sources)
    return body + sep + sources


def render_source(kb_id: str) -> None:
    """The page a citation link opens: the full knowledge-base entry."""
    doc = kb_entries().get(kb_id)
    if doc is None:
        st.error(f"There is no knowledge-base entry `{kb_id}`.")
        return
    question, _, answer = doc.page_content.partition("\nAnswer: ")
    st.caption(f"{kb_id[:2].upper()} knowledge base · {doc.metadata['category']} · `{kb_id}`")
    st.header(question.removeprefix("Question: "))
    st.markdown(answer)


# ── Page ────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Moxa — HR & IT Smart Assistant", page_icon=MOXA_ICON)
    # No config option hides only Deploy; toolbarMode would also drop Rerun and Clear cache.
    # The source-panel script's element is invisible, so don't let its slot push the page down.
    st.html(
        "<style>[data-testid='stAppDeployButton'],"
        " [data-testid='stElementContainer']:has(> [data-testid='stHtml'] script) { display: none; }</style>"
    )
    if source := st.query_params.get("source"):
        render_source(source)
        return
    # Sign-in is mandatory; source pages above stay open because the citation panel loads them in
    # their own (signed-out) session and they only show knowledge-base entries
    if not signed_in():
        render_landing()
        return
    st.html(SOURCE_PANEL, unsafe_allow_javascript=True)
    graph = get_graph()
    kb_loaded = knowledge_bases_ready()
    if "config" not in st.session_state:
        start_session()
    # Finish anything an interrupted run left behind
    resume_if_answered(graph)
    sync_replies(graph)

    render_sidebar(graph, kb_loaded)
    render_toolbar(graph, kb_loaded)
    # Sized to the title so the right-aligned caption ends where the title ends
    with st.container(width="content", gap=None):
        # Markdown h1 instead of st.title so the © can be raised with <sup>
        st.markdown("# Moxa<sup style='font-size: 0.5em'>©</sup> — HR & IT Smart Assistant", unsafe_allow_html=True, width="content")
        st.caption("Powered by LangGraph and Hybrid RAG", text_alignment="right")

    for message in st.session_state.history:
        with st.chat_message(message["role"], avatar=MOXA_ICON if message["role"] == "assistant" else EMPLOYEE_ICON):
            st.markdown(link_citations(message["content"]) if message["role"] == "assistant" else message["content"])

    # Ask about the first interrupt that still needs an answer
    unanswered = [item for item in pending_interrupts(graph) if item.id not in st.session_state.answers]
    if unanswered:
        render_interrupt(graph, unanswered[0])

    placeholder = "Reply to Moxa (yes / no, or extra details)" if unanswered else "Ask Moxa about HR or IT..."
    if text := st.chat_input(placeholder):
        if unanswered:
            answer_interrupt(graph, unanswered[0], text, shown=text)
        else:
            send_message(graph, text)
        st.rerun()


if __name__ == "__main__":
    main()
