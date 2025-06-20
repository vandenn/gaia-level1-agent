<!---
---
title: GAIA Level 1 Agent
emoji: 🕵🏻‍♂️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---
-->

<div align="center">

# GAIA Level 1 Agent

![python](https://img.shields.io/badge/python-3.11-blue)

This is an agent I built with [smolagents](https://huggingface.co/docs/smolagents/en/index) for solving [GAIA level 1 benchmark questions](https://huggingface.co/spaces/gaia-benchmark/leaderboard) as part of [HuggingFace's agents course](https://huggingface.co/learn/agents-course/unit0/onboarding).

</div>

# 📄 Overview

The app primarily runs on GradIO and smolagents.

# 🛠 Setup

Make sure you have `uv` installed in your system, then run:
```bash
make init
make run
```
If deploying to HuggingFace Spaces, make sure to also run the following to update the `requirements.txt` file, which is what is installed in the running space:
```bash
make sync-requirements
```
