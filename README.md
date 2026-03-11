# Mind the Gaps: Feedback-Driven Ergodic Coverage in Unknown Environments

Official implementation of the paper:

**Mind the Gaps: Feedback-Driven Ergodic Coverage in Unknown Environments**  
(ACC 2026)

This repository contains the implementation and simulation environment for **feedback-driven ergodic coverage in unknown environments**, enabling autonomous systems to discover and explore previously unknown regions while maintaining ergodic exploration properties.

---

## Overview

Exploration in unknown environments often requires balancing **coverage**, **information gathering**, and **adaptation to newly discovered structure**.

This work introduces a **feedback-driven ergodic coverage framework** that dynamically adapts the target spatial distribution based on newly discovered gaps in the environment.

The key idea is to:

1. Detect **unexplored regions (gaps)** in the environment.
2. Update the **ergodic target distribution** to emphasize these regions.
3. Drive the system using **ergodic control** so that trajectories proportionally cover the updated distribution.

This results in exploration strategies that:

- Adapt online to unknown environments
- Focus exploration on uncovered regions
- Maintain theoretical ergodic coverage guarantees

---

## Method Summary

Our approach combines three components:

### Gap Detection

A feedback mechanism identifies regions of the workspace that remain insufficiently explored.

### Adaptive Target Distribution

The spatial distribution used by ergodic control is updated to emphasize the detected gaps.

### Ergodic Control

The robot trajectory is driven to minimize the **ergodic metric**, ensuring time-averaged visitation statistics match the updated spatial distribution.

---

## Repository Structure


## Citation

If you use this code in your research, please cite:

@inproceedings{mindthegaps2026,\
title={Mind the Gaps: Feedback-Driven Ergodic Coverage in Unknown Environments},\
author={Thales C. Silva and Nora Ayanian},\
booktitle={American Control Conference (ACC)},\
year={2026}\
}
