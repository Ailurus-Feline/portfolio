package engine;

import action.Action;
import action.BasicAttack;
import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.ArrayList;
import java.util.List;

/**
 * Main battle engine.
 *
 * Handles game loop and round-based combat flow.
 */
public class BattleEngine {
    private final GameUI ui;
    private final TurnOrderStrategy turnOrderStrategy;

    public BattleEngine(GameUI ui) {
        this.ui = ui;
        this.turnOrderStrategy = new SpeedBasedTurnOrder();
    }

    /**
     * Starts the game loop.
     *
     * Repeats until player chooses to stop.
     */
    public void startGame() {
        boolean running = true;
        while (running) {
            Player player = ui.choosePlayer();
            ui.chooseItems(player);
            List<Combatant> enemies = ui.chooseLevel();

            int round = 1;

            while (player.isAlive() && hasAlive(enemies)) {
                ui.printRoundHeader(round);
                spawnBackupIfNeeded(ui, enemies);

                List<Combatant> turnOrder = buildTurnOrder(player, enemies);
                ui.displayBattleStatus(player, enemies);

                for (Combatant actor : turnOrder) {
                    if (!actor.isAlive()) {
                        continue;
                    }

                    actor.turnStart();

                    if (!player.isAlive() || !hasAlive(enemies)) {
                        break;
                    }

                    if (actor.isFrozen()) {
                        ui.printSkipTurn(actor);
                        actor.turnEnd();
                        continue;
                    }

                    if (actor instanceof Player) {
                        Action action = ui.chooseAction(player);
                        action.execute(actor, null, player, enemies, ui);
                    } else {
                        Action action = new BasicAttack();
                        action.execute(actor, player, player, enemies, ui);
                    }

                    actor.turnEnd();

                    if (!player.isAlive() || !hasAlive(enemies)) {
                        break;
                    }
                }

                round++;
            }

            ui.displayGameResult(player, enemies, round - 1);
            running = ui.askReplaySameSettings(player);
        }
    }

    /**
     * Builds turn order for current round.
     *
     * Only includes alive combatants.
     */
    private List<Combatant> buildTurnOrder(Player player, List<Combatant> enemies) {
        List<Combatant> combatants = new ArrayList<>();
        combatants.add(player);
        for (Combatant enemy : enemies) {
            if (enemy.isAlive()) {
                combatants.add(enemy);
            }
        }
        return turnOrderStrategy.whatOrder(combatants);
    }


    private boolean hasAlive(List<Combatant> combatants) {
        for (Combatant combatant : combatants) {
            if (combatant.isAlive()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Spawns backup enemies if conditions are met.
     */
    private void spawnBackupIfNeeded(GameUI ui, List<Combatant> enemies) {
        if (!ui.hasBackupSpawned() && !hasAlive(enemies)) {
            List<Combatant> backup = ui.spawnBackupEnemies();
            if (!backup.isEmpty()) {
                enemies.addAll(backup);
                ui.markBackupSpawned();
                ui.printBackupSpawn(backup);
            }
        }
    }
}