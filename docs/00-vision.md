# MultiHead Vision

## One-Liner Definition

**MultiHead is a local multimodal task-runner that decomposes work into steps and executes them by hot-swapping specialized models and tools in sequence, using RAG for memory and optionally calling online models when needed.**

## The Core Idea: Virtual Assembly Line

MultiHead is a local orchestration layer that lets one modest machine (even "one dinky GPU") behave like it has a whole multimodal lab installed -- by swapping different specialist models ("heads") in and out as needed.

Instead of one giant always-on model doing everything, you run:

- An **LLM head** (planning / decomposition / text reasoning)
- A **VLM head** (image understanding / verification)
- Maybe a **3D head** (mesh/scene generation or 3D processing)
- An **audio head** (TTS / voice / ASR)
- etc.

Because compute is limited, you don't run them all at once. You run them **serially**.

The "assembly line" is: **Model 1 -> Model 2 -> Model 3 -> ... -> done**, with artifacts passed between them.

This is how you get capability expansion on limited hardware: you're not trying to fit everything in VRAM simultaneously.

## Why It Matters: Ownership + Persistence

The power isn't just compute -- it's **ownership + persistence**:

- The system is on your machine
- Its memory/artifacts are yours
- Nobody can revoke access
- It can build up a durable working set over time (docs, notes, embeddings, prior outputs)
- Delete means delete. No ghost memory.

## Local + Optional Online Models

MultiHead can run:

- **Local heads** (your own models on your machine)
- **Remote heads** (API models online) if you want

It acts as a router + scheduler:

- Use local if good enough / cheaper / private
- Fall back to online if the step needs more power or a niche capability

## RAG / Vector DB: "Level Up" Without Training

You don't need to train a huge local LLM. You can use RAG / vector DB so the system becomes domain-aware by retrieving the right context at runtime. Your local LLM can be smaller/cheaper but still act smart because it pulls relevant memory/docs/code snippets.

## Why Multi-Model Beats One Bigger Model

A "factory line" of specialized ML tools + deterministic expert-system glue can beat a single giant frontier model on real work. Not because the big model is "dumber," but because **systems win**:

- You get specialists (OCR, detection, segmentation, retrieval, planners) instead of one generalist guessing
- You can verify every intermediate artifact (scores, constraints, invariants) instead of trusting vibes
- You can cache + reuse parts (expensive steps don't re-run)
- You can control memory + state explicitly (what's saved, what's deleted, what's canonical)
- You can do search / enumeration / retries / ensembling -- stuff a monolithic model can't do reliably in one shot

A frontier model is still useful -- but in this setup it becomes one stage (the expensive "judge / planner / last resort"), not the whole factory.

## What the "Heads" Look Like in Practice

A head is basically:

- A model (local or remote)
- An IO contract (what inputs it takes, what outputs it produces)
- A resource profile (VRAM/RAM/time)
- A runner (how to start/stop it)

Example head types:

| Head | Role |
|------|------|
| **LLM Planner** | Converts request -> step graph, picks tools/heads |
| **VLM Inspector** | Images -> captions, detections, verification |
| **Image Gen** | Storyboard frames, mockups |
| **3D** | Generate/clean meshes, layouts, or call a 3D toolchain |
| **Audio** | TTS narration, voice cloning (if allowed), audio cleanup |

## Example Workflow

User: "Take these screenshots + notes and make a narrated demo video."

MultiHead executes:

1. **LLM**: outline script + shot list
2. **VLM**: read screenshots, extract UI structure + key labels
3. **LLM**: refine script with extracted facts
4. **Image/Video head**: generate missing b-roll frames / transitions
5. **Audio head**: generate narration
6. **Tool head**: assemble into video (ffmpeg)
7. **LLM**: QA pass + fix timing notes

All serialized, artifact-driven.

## Design Principles

- **Local-first**: works on one laptop with no external deps
- **No marketplace required. No cloud required. No account required.**
- **Standalone**: friends can install it and run it on their own machines
- **Composable**: many small "workers" behind one uniform interface
- **Deterministic I/O**: strict schemas, artifact pointers, caching by hash
- **Deletable**: delete files + delete rows + rebuild optional index = real deletion

