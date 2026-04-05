package ui;

import action.Action;
import action.BasicAttack;
import action.DefendAction;
import action.SpecialSkillAction;
import action.UseItemAction;
import combat.Combatant;
import combat.Goblin;
import combat.Player;
import combat.Warrior;
import combat.Wizard;
import combat.Wolf;
import effect.StatusEffect;
import item.Item;
import item.Potion;
import item.PowerStone;
import item.SmokeBomb;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class GameUI {
    private final Scanner scanner;
    private int selectedLevel;
    private boolean backupSpawned;

    public GameUI() {
        this.scanner = new Scanner(System.in);
        this.selectedLevel = 1;
        this.backupSpawned = false;
    }

    public Player choosePlayer() {
        System.out.println("=== Turn-Based Combat Arena ===");
        System.out.println("1. Warrior (HP 260, ATK 40, DEF 20, SPD 30)");
        System.out.println("2. Wizard  (HP 200, ATK 50, DEF 10, SPD 20)");
        while (true) {
            System.out.print("Choose player: ");
            String input = scanner.nextLine();
            if ("1".equals(input)) {
                return new Warrior();
            }
            if ("2".equals(input)) {
                return new Wizard();
            }
            System.out.println("Invalid choice.");
        }
    }

    public void chooseItems(Player player) {
        System.out.println("Choose 2 items:");
        for (int i = 0; i < 2; i++) {
            player.addItem(createItemByChoice(promptItemChoice(i + 1)));
        }
    }

    public List<Combatant> chooseLevel() {
        backupSpawned = false;
        System.out.println("Choose difficulty:");
        System.out.println("1. Easy   (3 Goblins)");
        System.out.println("2. Medium (1 Goblin + 1 Wolf, backup 2 Wolves)");
        System.out.println("3. Hard   (2 Goblins, backup 1 Goblin + 2 Wolves)");
        while (true) {
            System.out.print("Select level: ");
            String input = scanner.nextLine();
            if ("1".equals(input) || "2".equals(input) || "3".equals(input)) {
                selectedLevel = Integer.parseInt(input);
                return spawnInitialEnemies();
            }
            System.out.println("Invalid choice.");
        }
    }

    private int promptItemChoice(int index) {
        System.out.println(index + ". Pick item:");
        System.out.println("1. Potion");
        System.out.println("2. Power Stone");
        System.out.println("3. Smoke Bomb");
        while (true) {
            System.out.print("Choice: ");
            String input = scanner.nextLine();
            if ("1".equals(input) || "2".equals(input) || "3".equals(input)) {
                return Integer.parseInt(input);
            }
            System.out.println("Invalid choice.");
        }
    }

    private Item createItemByChoice(int choice) {
        if (choice == 1) {
            return new Potion();
        }
        if (choice == 2) {
            return new PowerStone();
        }
        return new SmokeBomb();
    }

    private List<Combatant> spawnInitialEnemies() {
        List<Combatant> enemies = new ArrayList<>();
        if (selectedLevel == 1) {
            enemies.add(new Goblin());
            enemies.add(new Goblin());
            enemies.add(new Goblin());
        } else if (selectedLevel == 2) {
            enemies.add(new Goblin());
            enemies.add(new Wolf());
        } else {
            enemies.add(new Goblin());
            enemies.add(new Goblin());
        }
        return enemies;
    }

    public List<Combatant> spawnBackupEnemies() {
        List<Combatant> enemies = new ArrayList<>();
        if (selectedLevel == 2) {
            enemies.add(new Wolf());
            enemies.add(new Wolf());
        } else if (selectedLevel == 3) {
            enemies.add(new Goblin());
            enemies.add(new Wolf());
            enemies.add(new Wolf());
        }
        return enemies;
    }

    public boolean hasBackupSpawned() {
        return backupSpawned;
    }

    public void markBackupSpawned() {
        backupSpawned = true;
    }

    public Action chooseAction(Player player) {
        while (true) {
            System.out.println("Choose action:");
            System.out.println("1. BasicAttack");
            System.out.println("2. Defend");
            System.out.println("3. Item");
            System.out.println("4. SpecialSkill");
            System.out.print("Action: ");
            String input = scanner.nextLine();

            if ("1".equals(input)) {
                return new BasicAttack();
            }
            if ("2".equals(input)) {
                return new DefendAction();
            }
            if ("3".equals(input)) {
                return new UseItemAction();
            }
            if ("4".equals(input)) {
                return new SpecialSkillAction();
            }
            System.out.println("Invalid choice.");
        }
    }

    public Combatant chooseTarget(List<Combatant> enemies) {
        List<Combatant> alive = new ArrayList<>();
        for (Combatant enemy : enemies) {
            if (enemy.isAlive()) {
                alive.add(enemy);
            }
        }

        if (alive.isEmpty()) {
            return null;
        }

        System.out.println("Choose target:");
        for (int i = 0; i < alive.size(); i++) {
            Combatant enemy = alive.get(i);
            System.out.println((i + 1) + ". " + enemy.getName() + " HP: " + enemy.getHp() + "/" + enemy.getMaxHp());
        }

        while (true) {
            System.out.print("Target: ");
            String input = scanner.nextLine();
            try {
                int index = Integer.parseInt(input) - 1;
                if (index >= 0 && index < alive.size()) {
                    return alive.get(index);
                }
            } catch (NumberFormatException ignored) {
            }
            System.out.println("Invalid choice.");
        }
    }

    public Item chooseItem(Player player) {
        List<Item> items = player.getInventory();
        if (items.isEmpty()) {
            System.out.println("No items left.");
            return null;
        }

        System.out.println("Choose item:");
        for (int i = 0; i < items.size(); i++) {
            System.out.println((i + 1) + ". " + items.get(i).getName());
        }

        while (true) {
            System.out.print("Item: ");
            String input = scanner.nextLine();
            try {
                int index = Integer.parseInt(input) - 1;
                if (index >= 0 && index < items.size()) {
                    return items.get(index);
                }
            } catch (NumberFormatException ignored) {
            }
            System.out.println("Invalid choice.");
        }
    }

    public void displayBattleStatus(Player player, List<Combatant> enemies) {
        System.out.println();
        System.out.println("Player: " + player.getName() + " HP " + player.getHp() + "/" + player.getMaxHp()
                + " ATK " + player.getAttack()
                + " DEF " + player.getEffectiveDefense()
                + " SPD " + player.getSpeed()
                + " CD " + player.getCooldown());
        printEffects(player);

        System.out.println("Enemies:");
        for (Combatant enemy : enemies) {
            System.out.println("- " + enemy.getName() + " HP " + enemy.getHp() + "/" + enemy.getMaxHp()
                    + (enemy.isAlive() ? "" : " [ELIMINATED]"));
            printEffects(enemy);
        }

        System.out.println("Items:");
        if (player.getInventory().isEmpty()) {
            System.out.println("- None");
        } else {
            for (Item item : player.getInventory()) {
                System.out.println("- " + item.getName());
            }
        }
        System.out.println();
    }

    private void printEffects(Combatant combatant) {
        List<StatusEffect> effects = combatant.getStatusEffects();
        if (effects.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder("  Effects: ");
        for (int i = 0; i < effects.size(); i++) {
            StatusEffect effect = effects.get(i);
            sb.append(effect.getName()).append("(").append(effect.getDuration()).append(")");
            if (i < effects.size() - 1) {
                sb.append(", ");
            }
        }
        System.out.println(sb);
    }

    public void printRoundHeader(int round) {
        System.out.println("========== Round " + round + " ==========");
    }

    public void printAttack(Combatant actor, Combatant target, int damage) {
        System.out.println(actor.getName() + " attacks " + target.getName() + " for " + damage + " damage. "
                + target.getName() + " HP: " + target.getHp() + "/" + target.getMaxHp());
    }

    public void printDefend(Combatant actor) {
        System.out.println(actor.getName() + " uses Defend. Defense +10 for current and next turn.");
    }

    public void printShieldBash(Player player, Combatant target, int damage) {
        System.out.println(player.getName() + " uses Shield Bash on " + target.getName()
                + " for " + damage + " damage and applies Stun.");
    }

    public void printArcaneBlast(Player player) {
        System.out.println(player.getName() + " uses Arcane Blast on all enemies.");
    }

    public void printWizardAttackBoost(Player player) {
        System.out.println(player.getName() + " gains +10 attack from Arcane Blast kill. Current ATK: " + player.getAttack());
    }

    public void printPotionUsed(Player player, int before, int after) {
        System.out.println(player.getName() + " uses Potion. HP: " + before + " -> " + after);
    }

    public void printPowerStoneUsed(Player player) {
        System.out.println(player.getName() + " uses Power Stone. Special skill triggered without changing cooldown.");
    }

    public void printSmokeBombUsed(Player player) {
        System.out.println(player.getName() + " uses Smoke Bomb. Enemy attacks deal 0 damage this turn and next turn.");
    }

    public void printNoItemUsed() {
        System.out.println("Item action cancelled.");
    }

    public void printSkillOnCooldown(Player player) {
        System.out.println("Special skill is on cooldown for " + player.getCooldown() + " more turn(s).");
    }

    public void printSkipTurn(Combatant actor) {
        System.out.println(actor.getName() + " is unable to act and skips the turn.");
    }

    public void printInvulnerable(Combatant target) {
        System.out.println(target.getName() + " is invulnerable. Damage becomes 0.");
    }

    public void printBackupSpawn(List<Combatant> backup) {
        System.out.println("Backup Spawn triggered:");
        for (Combatant enemy : backup) {
            System.out.println("- " + enemy.getName());
        }
    }

    public void displayGameResult(Player player, List<Combatant> enemies, int rounds) {
        System.out.println();
        if (player.isAlive()) {
            System.out.println("Victory! Congratulations, you have defeated all your enemies.");
            System.out.println("Remaining HP: " + player.getHp() + "/" + player.getMaxHp());
            System.out.println("Total Rounds: " + rounds);
        } else {
            int enemiesRemaining = 0;
            for (Combatant enemy : enemies) {
                if (enemy.isAlive()) {
                    enemiesRemaining++;
                }
            }
            System.out.println("Defeated. Don't give up, try again!");
            System.out.println("Enemies Remaining: " + enemiesRemaining);
            System.out.println("Total Rounds Survived: " + rounds);
        }
        System.out.println();
    }

    public boolean askReplaySameSettings(Player previousPlayer) {
        while (true) {
            System.out.println("1. Replay");
            System.out.println("2. Exit");
            System.out.print("Choice: ");
            String input = scanner.nextLine();
            if ("1".equals(input)) {
                return true;
            }
            if ("2".equals(input)) {
                return false;
            }
            System.out.println("Invalid choice.");
        }
    }
}
