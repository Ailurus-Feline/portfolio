package action;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

public interface CooldownAction extends Action {
    void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui, boolean ifCooldown);

    @Override
    default void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui) {
        execute(actor, directTarget, player, enemies, ui, true);
    }
    String getName();
}