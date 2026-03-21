# Agent System

The autonomous research system runs experiments, evaluates results, and accumulates knowledge — without a human in the loop for each cycle. This page describes the agents, skills, and knowledge architecture that make that possible.

## The Experiment Cycle

Every experiment follows the same lifecycle. The conductor orchestrates, specialized agents handle each phase, and knowledge flows back into the system for the next cycle.

```mermaid
flowchart LR
    subgraph PROPOSE["1 · Propose"]
        H[Hypothesis Agent]
    end

    subgraph REVIEW["2 · Review"]
        D[Research Director]
    end

    subgraph BUILD["3 · Build"]
        C[Coder Agent]
    end

    subgraph TRAIN["4 · Train"]
        M[Modal / GPU]
    end

    subgraph EVALUATE["5 · Evaluate"]
        D2[Research Director]
    end

    subgraph LEARN["6 · Learn"]
        K[Knowledge Base]
    end

    H -->|hypothesis| D
    D -->|approved| C
    D -->|rejected| K
    C -->|branch + config| M
    M -->|metrics| D2
    D2 -->|kept / discarded| K
    K -.->|informs| H

    style PROPOSE fill:#1565c0,color:#fff
    style REVIEW fill:#4a148c,color:#fff
    style BUILD fill:#2e7d32,color:#fff
    style TRAIN fill:#e65100,color:#fff
    style EVALUATE fill:#4a148c,color:#fff
    style LEARN fill:#bf360c,color:#fff
```

The conductor (`/conductor`) runs this cycle on a timer — typically every 60 minutes. Each heartbeat checks in-flight experiments, evaluates any that finished, and launches new ones if budget and slots allow.

## Agents

Each agent has a single responsibility. No agent both proposes and evaluates. This separation prevents the system from confirming its own biases.

```mermaid
flowchart TD
    subgraph CONDUCTOR["Conductor"]
        CO[Orchestrator]
    end

    subgraph GENERATORS["Generators"]
        HY[Hypothesis Agent]
        RE[Researcher]
        SP[Spec Agent]
    end

    subgraph GATES["Quality Gates"]
        RD[Research Director]
        RV[Reviewer]
    end

    subgraph EXECUTORS["Executors"]
        CD[Coder Agent]
    end

    CO -->|spawns| HY
    CO -->|spawns| RD
    CO -->|spawns| CD
    HY -.->|proposes to| RD
    CD -.->|reviewed by| RV
    RE -.->|findings to| HY
    SP -.->|briefs to| CD

    style CONDUCTOR fill:#4a148c,color:#fff
    style GENERATORS fill:#1565c0,color:#fff
    style GATES fill:#b71c1c,color:#fff
    style EXECUTORS fill:#2e7d32,color:#fff
```

| Agent | Role | Reads | Produces |
|-------|------|-------|----------|
| **Conductor** | Orchestrator. Runs the heartbeat, manages slots and budget, spawns other agents. | State files, budget, in-flight experiments | Decisions, Matrix notifications |
| **Hypothesis Agent** | Proposes one experiment per cycle. Grounds proposals in evidence and literature. | program.md, run cards, decisions, papers | Structured hypothesis + draft run card |
| **Research Director** | Quality gate. Approves/rejects hypotheses, evaluates completed experiments. | program.md, run cards, budget, metrics | APPROVE/REJECT with reasoning, KEPT/DISCARDED verdicts |
| **Coder** | Implements approved experiments. Works in isolated git worktrees. | Approved hypothesis, base config, model code | Experiment branch, config YAML, run card |
| **Researcher** | Investigates issues. Reads docs, explores code, writes findings. Never modifies code. | Anything in the repo or on the web | Findings in `.loop/knowledge/` |
| **Reviewer** | Code review on completed work. Checks quality, scope, correctness. | Branch diffs, completion criteria | APPROVE / REQUEST CHANGES verdict |
| **Spec Agent** | Helps scope work into structured briefs that coders can execute autonomously. | Human intent, project context | Briefs with tasks and completion criteria |

## Knowledge Architecture

Knowledge lives in three layers. Agents don't carry all knowledge in context — they call **consult skills** to access domain expertise on demand.

```mermaid
flowchart TD
    subgraph AGENTS["Agents"]
        HY2[Hypothesis]
        RD2[Director]
        CD2[Coder]
    end

    subgraph SKILLS["Consult Skills"]
        CE["/consult-empirical"]
        CC["/consult-canon"]
        CA["/consult-architecture"]
    end

    subgraph KNOWLEDGE["Knowledge Base · .loop/knowledge/"]
        LE["learnings.md\n— empirical findings, hit rates"]
        WM["world-model-patterns.md\n— literature patterns"]
        M2["mamba2-properties.md\n— architecture properties"]
    end

    subgraph SOURCES["Source Material"]
        RC2["docs/run-cards/\n— experiment records"]
        PM["program.md\n— research directions"]
        RS["research/sources/\n— paper summaries"]
        RCN["docs/research-canon.md\n— literature index"]
        DEC["docs/decisions/\n— rejected hypotheses"]
    end

    HY2 -->|"what's been tried?"| CE
    HY2 -->|"prior art?"| CC
    RD2 -->|"is this grounded?"| CE
    CD2 -->|"how does Mamba2 handle this?"| CA

    CE --> LE
    CE --> RC2
    CE --> DEC
    CC --> WM
    CC --> RS
    CC --> RCN
    CA --> M2
    CA --> PM

    RC2 -.->|"experiments update"| LE
    RS -.->|"papers update"| WM

    style AGENTS fill:#1565c0,color:#fff
    style SKILLS fill:#4a148c,color:#fff
    style KNOWLEDGE fill:#bf360c,color:#fff
    style SOURCES fill:#37474f,color:#fff
```

### Consult Skills

Skills are active capabilities, not passive documents. An agent invokes a skill with a question and gets a grounded answer — the skill loads its knowledge base, reads current state, and reasons about the specific question.

| Skill | When to Call | What It Knows |
|-------|-------------|---------------|
| `/consult-empirical` | Proposing an experiment and need to check if it's been tried. Evaluating results against prior findings. | Run cards, decisions, learnings. Hit rates and evidence density. |
| `/consult-canon` | Evaluating prior art. Seeking techniques from adjacent paradigms. Checking if an approach has been studied. | Research canon, paper summaries, world model patterns. |
| `/consult-architecture` | Experiment touches model structure, training regime, or loss design. Need to understand Mamba2 constraints. | Mamba2 properties, model code, training code, VRAM estimates. |

### Knowledge Base

The knowledge base (`.loop/knowledge/`) is the system's institutional memory — distilled findings that persist across sessions. It is **not** a raw dump. Each file is curated to be context-sized (2-3 pages) and actionable for experiment design.

| File | Contents | Updated By |
|------|----------|------------|
| `learnings.md` | Empirical findings with hit rates and experiment citations | Director (after evaluating results) |
| `mamba2-properties.md` | Architecture properties that matter for experiment design | Manual curation + researcher findings |
| `world-model-patterns.md` | Literature patterns mapped to our project | Research loop + manual curation |

### Practices Across All Skills

These principles apply to every consult skill:

- **Flag uncited assumptions.** If a claim has no experiment ID or paper reference, it may come from the LLM's training data. Name it: "This claim has no citation. Verify before building on it."
- **Surface divergences.** We use an SSM (not transformer), predict structured state (not pixels), train on replays (not video). Name where we diverge from the mainstream and whether it matters.
- **Track paradigm shifts.** Situate our work in the evolution of the field, not just the current snapshot.
- **Push knowledge forward.** Surface tensions and open questions. The goal is collective understanding, not confirmation.

## The Learning Loop

The system's value isn't any single experiment — it's the accumulation of knowledge over many cycles. Each cycle produces findings that change what the next cycle proposes.

```mermaid
flowchart TD
    EXP["Run Experiment"] --> EVAL["Evaluate Results"]
    EVAL --> RC["Update Run Card\n(status, RC, findings)"]
    RC --> KB["Update Knowledge Base\n(learnings.md, hit rates)"]
    KB --> NEXT["Next Hypothesis"]
    NEXT --> CONSULT["Consult Skills\n(empirical, canon, architecture)"]
    CONSULT --> GROUND["Grounded Proposal"]
    GROUND --> DIR["Director Review"]
    DIR -->|approved| EXP
    DIR -->|rejected| DEC2["Log Decision\n(docs/decisions/)"]
    DEC2 -.-> NEXT

    style EXP fill:#e65100,color:#fff
    style EVAL fill:#4a148c,color:#fff
    style RC fill:#37474f,color:#fff
    style KB fill:#bf360c,color:#fff
    style NEXT fill:#1565c0,color:#fff
    style CONSULT fill:#4a148c,color:#fff
    style GROUND fill:#1565c0,color:#fff
    style DIR fill:#b71c1c,color:#fff
    style DEC2 fill:#37474f,color:#fff
```

**What makes this a loop, not a pipeline:**

- **Rejected hypotheses are knowledge too.** They're logged in `docs/decisions/` with full Director reasoning. Future hypothesis agents read these — they know what was considered and why it was turned down.
- **Knowledge compounds.** The empirical findings table grows with each experiment. Hit rates become more reliable. Open axes narrow or expand based on results.
- **The system can surprise itself.** An experiment proposed for one reason might reveal something unexpected. The Director captures this in the evaluation, and it enters the knowledge base as a new finding.
- **Humans stay in the loop on direction.** `program.md` is the human's lever. The system proposes and tests within the bounds that `program.md` sets. Changing `program.md` redirects the entire research frontier.

## State and Budget

The conductor tracks operational state to prevent runaway spending and manage concurrency.

| File | Purpose |
|------|---------|
| `.loop/state/budget.json` | Daily/weekly spend limits and tracking |
| `.loop/state/running.json` | In-flight experiments (up to 3 concurrent) |
| `.loop/state/log.jsonl` | Append-only decision log |
| `.loop/state/signals/pause.json` | Pause signal — stops all new launches |
| `.loop/state/signals/escalate.json` | Escalation — surfaces to human for decision |
