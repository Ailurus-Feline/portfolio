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

/**
 * Game user interface.
 *
 * Responsible for:
 * - player input
 * - battle information display
 * - remembering previous game settings
 * - loading screen display
 */
public class GameUI {
    private final Scanner scanner;
    private int selectedLevel;
    private boolean backupSpawned;

    private Player lastPlayerTemplate;
    private List<Item> lastItems;
    private int lastLevel;

    /**
     * Replay menu choice after battle.
     *
     * 1 = replay with same settings
     * 2 = start with new settings
     * 3 = exit
     */
    private int replayChoice;

    public GameUI() {
        this.scanner = new Scanner(System.in);
        this.selectedLevel = 1;
        this.backupSpawned = false;
        this.lastPlayerTemplate = null;
        this.lastItems = new ArrayList<>();
        this.lastLevel = 1;
        this.replayChoice = 3;
    }

    /**
     * Saves the chosen class as replay template.
     */
    public Player choosePlayer() {
        System.out.println("=== Turn-Based Combat Arena ===");
        System.out.println("1. Warrior (HP 260, ATK 40, DEF 20, SPD 30)");
        System.out.println("2. Wizard  (HP 200, ATK 50, DEF 10, SPD 20)");
        while (true) {
            System.out.print("Choose player: ");
            String input = scanner.nextLine();
            System.out.println();

            if ("1".equals(input)) {
                lastPlayerTemplate = new Warrior();
                return new Warrior();
            }
            if ("2".equals(input)) {
                lastPlayerTemplate = new Wizard();
                return new Wizard();
            }

            System.out.println("Invalid choice.\n");
            System.out.println();
        }
    }

    public void chooseItems(Player player) {
        lastItems = new ArrayList<>();
        System.out.println("Choose 2 items:");
        for (int i = 0; i < 2; i++) {
            Item item = createItemByChoice(itemChoice(i + 1));
            player.addItem(item);
            lastItems.add(copyItem(item));
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
            System.out.println();

            if ("1".equals(input) || "2".equals(input) || "3".equals(input)) {
                selectedLevel = Integer.parseInt(input);
                lastLevel = selectedLevel;
                List<Combatant> enemies = spawnInitialEnemies();
                printLoadingScreen(enemies);
                return enemies;
            }

            System.out.println("Invalid choice.");
            System.out.println();
        }
    }

    /**
     * Prints loading screen before battle starts.
     *
     * Shows enemy list and their attributes.
     */
    public void printLoadingScreen(List<Combatant> enemies) {
        System.out.println();
        System.out.println("========== Loading Battle ==========");
        System.out.println("Selected Difficulty: " + getLevelName(selectedLevel));
        System.out.println("Enemies:");
        for (Combatant enemy : enemies) {
            printCombatantAttributes(enemy);
        }
        System.out.println("====================================");
        System.out.println();
        parse("start the battle!");
    }

    public boolean ifReplaySameSettings() {
        return replayChoice == 1;
    }

    public boolean ifStartNewSettings() {
        return replayChoice == 2;
    }

    public boolean ifExit() {
        return replayChoice == 3;
    }

    /**
     * Builds a fresh player using saved replay settings.
     */
    public Player createReplayPlayer() {
        Player player = copyPlayer(lastPlayerTemplate);
        for (Item item : lastItems) {
            player.addItem(copyItem(item));
        }
        return player;
    }

    /**
     * Builds fresh enemies using saved replay level.
     */
    public List<Combatant> createReplayEnemies() {
        selectedLevel = lastLevel;
        backupSpawned = false;
        List<Combatant> enemies = spawnInitialEnemies();
        printLoadingScreen(enemies);
        return enemies;
    }

    /**
     * Asks the player to choose an item.
     */
    private int itemChoice(int index) {
        System.out.println(index + ". Pick item:");
        System.out.println("\t1. Potion");
        System.out.println("\t2. Power Stone");
        System.out.println("\t3. Smoke Bomb");
        while (true) {
            System.out.print("Choice: ");
            String input = scanner.nextLine();
            System.out.println();

            if ("1".equals(input) || "2".equals(input) || "3".equals(input)) {
                return Integer.parseInt(input);
            }
            System.out.println("Invalid choice.");
            System.out.println();
        }
    }

    /**
     * Creates an item instance based on choice number.
     */
    private Item createItemByChoice(int choice) {
        if (choice == 1) {
            return new Potion();
        }
        if (choice == 2) {
            return new PowerStone();
        }
        return new SmokeBomb();
    }

    /**
     * Creates initial enemies for the current selected level.
     */
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

    /**
     * Spawns backup enemies for the current selected level.
     */
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

    /**
     * Marks that backup enemies have been spawned.
     */
    public void markBackupSpawned() {
        backupSpawned = true;
    }

    /**
     * Lets the player choose an action.
     */
    public Action chooseAction(Player player) {
        while (true) {
            System.out.println();
            System.out.println("Choose action:");
            System.out.println("1. BasicAttack");
            System.out.println("2. Defend");
            System.out.println("3. Item");
            System.out.println("4. SpecialSkill");
            System.out.print("Action: ");
            String input = scanner.nextLine();
            System.out.println();

            if ("1".equals(input)) {
                return new BasicAttack();
            }
            if ("2".equals(input)) {
                return new DefendAction();
            }
            if ("3".equals(input)) {
                if (player.getInventory().isEmpty()) {
                    printNoItem();
                    continue;
                }
                return new UseItemAction();
            }
            if ("4".equals(input)) {
                if (player.getCooldown() > 0) {
                    printSkillOnCooldown(player);
                    System.out.println();
                    continue;
                }
                return new SpecialSkillAction();
            }

            System.out.println("Invalid choice.");
            System.out.println();
        }
    }

    /**
     * Lets the player choose a target from alive enemies.
     */
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
            System.out.println();

            try {
                int index = Integer.parseInt(input) - 1;
                if (index >= 0 && index < alive.size()) {
                    return alive.get(index);
                }
            } catch (NumberFormatException ignored) {
            }

            System.out.println("Invalid choice.");
            System.out.println();
        }
    }

    /**
     * Lets the player choose an item from inventory.
     */
    public Item chooseItem(Player player) {
        List<Item> items = player.getInventory();

        System.out.println("Choose item:");
        for (int i = 0; i < items.size(); i++) {
            System.out.println((i + 1) + ". " + items.get(i).getName());
        }

        while (true) {
            System.out.print("Item: ");
            String input = scanner.nextLine();
            System.out.println();

            try {
                int index = Integer.parseInt(input) - 1;
                if (index >= 0 && index < items.size()) {
                    return items.get(index);
                }
            } catch (NumberFormatException ignored) {
            }

            System.out.println("Invalid choice.");
            System.out.println();
        }
    }

    /**
     * Displays current battle status.
     *
     * Includes detailed enemy attributes.
     */
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
            System.out.println("- " + enemy.getName()
                    + " HP " + enemy.getHp() + "/" + enemy.getMaxHp()
                    + " ATK " + enemy.getAttack()
                    + " DEF " + enemy.getEffectiveDefense()
                    + " SPD " + enemy.getSpeed()
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

    /**
     * Prints active status effects of a combatant.
     */
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

    public void printNoItem() {
        System.out.println("No items available. Choose again.\n");
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

    /**
     * Shows post-game options.
     *
     * 1. Replay with same settings
     * 2. Start with new settings
     * 3. Exit
     *
     * Returns true if the game should continue.
     */
    public boolean askReplaySameSettings(Player previousPlayer) {
        while (true) {
            System.out.println("1. Replay with same settings");
            System.out.println("2. Start with new settings");
            System.out.println("3. Exit");
            System.out.print("Choice: ");
            String input = scanner.nextLine();

            if ("1".equals(input)) {
                replayChoice = 1;
                return true;
            }
            if ("2".equals(input)) {
                replayChoice = 2;
                return true;
            }
            if ("3".equals(input)) {
                replayChoice = 3;
                return false;
            }
            System.out.println("Invalid choice.");
            System.out.println(); 
        }
    }

    /**
     * Creates a copy of a player based on class type.
     */
    private Player copyPlayer(Player player) {
        if (player instanceof Warrior) {
            return new Warrior();
        }
        return new Wizard();
    }

    /**
     * Creates a fresh copy of an item based on item type.
     */
    private Item copyItem(Item item) {
        if (item instanceof Potion) {
            return new Potion();
        }
        if (item instanceof PowerStone) {
            return new PowerStone();
        }
        return new SmokeBomb();
    }

    private String getLevelName(int level) {
        if (level == 1) {
            return "Easy";
        }
        if (level == 2) {
            return "Medium";
        }
        return "Hard";
    }

    /**
     * Prints full combatant attributes.
     */
    private void printCombatantAttributes(Combatant combatant) {
        System.out.println("- " + combatant.getName()
                + " HP " + combatant.getHp() + "/" + combatant.getMaxHp()
                + " ATK " + combatant.getAttack()
                + " DEF " + combatant.getEffectiveDefense()
                + " SPD " + combatant.getSpeed());
    }

    public void parse(String target) {
        System.out.println("Press any key to " + target);
        scanner.nextLine();
        System.out.println();
    }
}