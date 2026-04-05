package action;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

public interface Action {
    void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui);
    String getName();
}
