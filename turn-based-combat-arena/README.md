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
- Speed-based turn order
- Multiple enemy waves (including backup spawn)
- Clear win/loss conditions (no draw scenario)

### Player Types
- **Warrior**
  - High HP and defense  
  - Special Skill: *Shield Bash* (damage + stun)

- **Wizard**
  - High attack  
  - Special Skill: *Arcane Blast* (AOE + scaling attack)

### Enemy Types
- Goblin
- Wolf

### Actions
- Basic Attack
- Defend
- Use Item
- Special Skill (with cooldown)

### Items
- Potion (healing)
- Power Stone (free skill usage)
- Smoke Bomb (temporary invulnerability)

### Status Effects
- Stun
- Defense Buff
- Smoke Bomb Invulnerability
- Arcane Scaling Buff

---

## Game Flow

1. Player selects:
   - Character (Warrior / Wizard)
   - Items
   - Difficulty level

2. Game loop:
   - Determine turn order (based on speed)
   - Apply status effects
   - Execute actions
   - Update HP and states
   - Check win/loss conditions

3. Game ends when:
   - Player HP reaches 0 → Defeat
   - All enemies defeated → Victory

---

## Architecture

```
combat/
├── engine/
├── character/
├── action/
├── item/
├── effect/
```

### Key Components

| Component            | Responsibility |
|---------------------|--------------|
| BattleEngine        | Controls battle flow |
| TurnOrderStrategy   | Determines turn sequence |
| Combatant           | Base abstraction |
| Player / Enemy      | Combat roles |
| Action              | Behaviors |
| StatusEffect        | Persistent effects |
| Item                | Consumables |

---

## Design Principles (SOLID)

- SRP: Each class has one responsibility  
- OCP: Extend without modifying core logic  
- LSP: Player/Enemy interchangeable  
- ISP: Small interfaces  
- DIP: Depend on abstractions  

---

## Design Patterns

- Strategy Pattern (Turn order)  
- Polymorphism (Combatant behavior)  
- Composition (Status effects)  

---

## How to Run

Compile:
javac -d out src/combat/Main.java

Run:
java -cp out combat.Main

---

## Extensibility

- Add new characters
- Add new actions
- Add new items
- Replace turn strategy

---

## Notes

- CLI-based implementation  
- Focus on software architecture  

---

## Author

Course Project (Software Design & OOP)
