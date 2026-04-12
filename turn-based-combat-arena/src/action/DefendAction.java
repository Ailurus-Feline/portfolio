package action;

import combat.Combatant;
import combat.Player;
import effect.DefendEffect;
import ui.GameUI;

import java.util.List;

/**
 * Temporarily increases actor's defense.
 *
 * Applies a DefendEffect lasting for 2 turns.
 */
public class DefendAction implements Action {

    @Override
    public void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui) {

        actor.addStatusEffect(new DefendEffect(2));

        ui.printDefend(actor);
    }

    @Override
    public String getName() {
        return "Defend";
    }
}