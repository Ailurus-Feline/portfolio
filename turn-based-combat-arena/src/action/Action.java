package action;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

/**
 * Action interface for all combat actions.
 * Each action defines how it is executed during a turn.
 */
public interface Action {

    /**
     * Execute the action.
     *
     * Args:
     *   actor
     *   directTarget
     *   player: player instance
     *   enemies
     *   ui: UI handler for interaction/output
     */
    void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui);

    String getName();
}