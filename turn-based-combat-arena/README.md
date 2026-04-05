# Turn-Based Combat Arena

## Overview
This project implements a **turn-based combat arena game** in Java, designed as a **course-based software engineering project** focusing on **Object-Oriented Programming (OOP)** and **SOLID principles**.

The system runs entirely on a **Command Line Interface (CLI)** and emphasizes:
- Extensibility
- Maintainability
- Clean architecture design

Players control a character to battle enemies using actions, items, and status effects until one side is defeated.

---

## Features

### Core Gameplay
- Turn-based combat system
- **Speed-based turn order (higher speed acts first)**
- Multiple enemy waves (**with conditional backup spawn**)
- Clear win/loss conditions (no draw scenario)
- Status effects processed at **turn start**

### Player Types
- **Warrior**
  - High HP and defense  
  - Special Skill: *Shield Bash* (damage + stun for 2 turns)

- **Wizard**
  - High attack  
  - Special Skill: *Arcane Blast* (AOE + scaling attack on kill)

### Enemy Types
- Goblin
- Wolf

### Actions
- Basic Attack (damage = attack - effective defense)
- Defend (+10 defense for 2 turns)
- Use Item (consumes item after use)
- Special Skill (with 3-turn cooldown)

### Items
- Potion (healing +100 HP)
- Power Stone (free skill usage without cooldown consumption)
- Smoke Bomb (temporary invulnerability for 2 turns)

### Status Effects
- Stun (skip turn)
- Defense Buff (+10 defense)
- Smoke Bomb Invulnerability (immune to damage)
- Arcane Scaling Buff (permanent attack increase on kill)

---

## Game Flow

1. Player selects:
   - Character (Warrior / Wizard)
   - Items (2 selections)
   - Difficulty level

2. Game loop:
   - Determine turn order (based on speed)
   - Apply status effects
   - Execute actions
   - Update HP and states
   - Reduce cooldowns
   - Check win/loss conditions

3. Game ends when:
   - Player HP reaches 0 → Defeat
   - All enemies defeated → Victory

---

## Project Structure

    turn-based-combat-arena/
    ├── README.md
    ├── .gitignore
    │
    ├── src/
    │   ├── Main.java
    │   │
    │   ├── engine/
    │   ├── combat/
    │   ├── action/
    │   ├── effect/
    │   ├── item/
    │   └── ui/
    │
    └── docs/
        ├── uml_class_diagram.png
        └── uml_sequence_diagram.png

---

## Architecture

Layered Design:
UI (CLI) → Engine → Domain (Combat / Action / Effect / Item)