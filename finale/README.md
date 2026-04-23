---
title: TriageAI - Emergency Room Crisis Simulator
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
license: mit
---

# 🏥 TriageAI — Emergency Room Crisis Simulator

> **Theme 5: Wild Card — Impress Us!**  
> *OpenEnv Hackathon India 2026 | Team Block Dragon*

**10 patients. 3 beds. 2 doctors. 1 operating room. The clock is ticking. Can an AI learn to save them all?**

---

## The Problem

Every day, emergency rooms face a critical challenge: too many patients, too few resources. Triage — the process of deciding who gets treated first — is one of the highest-stakes decision-making tasks in medicine. A wrong priority call can cost a life.

Current LLMs struggle with this type of problem because it requires:
- **Partial observability** — you can't see a patient's true condition without examining them
- **Resource allocation under constraints** — 3 beds, 2 doctors, 1 OR
- **Temporal reasoning** — untreated patients deteriorate and can die
- **Multi-step planning** — triage → assign bed → examine → treat → discharge → repeat

**TriageAI** is an OpenEnv-compliant RL environment that trains LLMs to perform emergency medical triage under pressure.

---

## How It Works

### Environment Loop

```
RESET → Generate ER scenario (patients with hidden severity)
  ↓
OBSERVE → See symptoms, vitals, hospital state (NOT hidden severity)
  ↓
ACT → Choose one of 8 actions
  ↓
REWARD → Multi-component score (survival, triage accuracy, etc.)
  ↓
REPEAT until all patients resolved or max steps reached
```

### Actions (8 total)

| Action | What It Does |
|--------|-------------|
| `triage` | Assess patient — reveals estimated severity level |
| `assign_bed` | Move patient from waiting to a bed |
| `assign_doctor` | Send doctor to examine (reveals clinical notes) |
| `order_treatment` | Prescribe medication/IV/oxygen/monitoring |
| `send_to_or` | Emergency surgery (patient must be in bed, OR must be free) |
| `discharge` | Discharge stable patients to free beds |
| `reassess` | Re-check vitals (useful as patients deteriorate) |
| `submit` | End episode for final scoring |

### Key Design: Partial Observability

The agent sees **symptoms and vitals** but NOT the hidden severity or diagnosis. It must:
1. **Triage** to get an estimated severity (with noise)
2. **Assign a doctor** to examine and get clinical notes
3. Only then make informed treatment decisions

This forces genuine **information gathering** — not just pattern matching.

### Patient Deterioration

Untreated patients **worsen over time**:
- Severity 5 (stable) → 4 → 3 → 2 → 1 (critical) → **DEAD**
- Critical patients deteriorate faster (2-3 steps)
- Stable patients are safe for longer (15-20 steps)

### 3 Difficulty Levels (Curriculum)

| Task | Patients | Beds | Docs | OR | Max Steps | Challenge |
|------|----------|------|------|----|-----------|-----------|
| `task_easy` | 4 | 3 | 2 | 1 | 20 | Clear symptoms |
| `task_medium` | 7 | 3 | 2 | 1 | 30 | Ambiguous cases, hidden critical |
| `task_hard` | 10 | 3 | 2 | 1 | 40 | Mass casualty, rapid deterioration |

---

## Reward Function (6 Components)

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **Survival Rate** | 35% | % of patients alive at end |
| **Triage Accuracy** | 20% | How close agent's triage level matches true severity |
| **Treatment Quality** | 15% | Did surgical patients get surgery? Did minor cases get discharged? |
| **Time Efficiency** | 15% | Were critical patients treated quickly? |
| **Resource Utilization** | 10% | Were beds/doctors used efficiently? |
| **Efficiency Bonus** | 5% | Did agent take meaningful actions (>3 steps)? |

**Anti-gaming protections:**
- Can't discharge critically ill patients
- Can't triage the same patient >2 times without penalty
- Can't send everyone to OR (only 1 OR, 3-step cooldown)
- Dead patients can't be treated
- Invalid actions return negative reward

---

## Quick Start

### API Endpoints

```bash
# Health check
curl https://hinex-07-triage-ai-env.hf.space/health

# Reset environment
curl -X POST https://hinex-07-triage-ai-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "seed": 42}'

# Take an action
curl -X POST https://hinex-07-triage-ai-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "triage", "patient_id": "P001"}'

# Check state
curl https://hinex-07-triage-ai-env.hf.space/state
```

### Run Locally

```bash
cd finale
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## Training

**Model:** Qwen2.5-3B-Instruct (with Unsloth 4-bit QLoRA)  
**Algorithm:** GRPO via TRL  
**Notebook:** [Training Colab Notebook](link-to-colab)

### Training Results

*Training plots will be added after onsite training on April 25-26.*

---

## Links

- **HF Space:** https://huggingface.co/spaces/hinex-07/triage-ai-env
- **GitHub:** https://github.com/hinex-vaghadiya/openenv-data-cleaning
- **Training Notebook:** [Colab](link-to-colab)
- **Blog/Video:** [Coming soon]

---

## Technical Stack

- **Framework:** OpenEnv (latest)
- **Server:** FastAPI + Uvicorn
- **Training:** TRL (GRPO) + Unsloth
- **Model:** Qwen2.5-3B-Instruct
- **Deployment:** Docker on HF Spaces

---

*Built with ❤️ by Team Block Dragon for OpenEnv Hackathon India 2026*
