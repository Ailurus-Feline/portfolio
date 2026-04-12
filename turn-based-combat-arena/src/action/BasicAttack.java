package action;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

/**
 * Basic single-target attack action.
 *
 * Damage = max(0, attack - defense).
 * Supports manual target selection.
 */
public class BasicAttack implements Action {

    @Override
    public void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui) {

        // Resolve target: use provided target or manually choose
        Combatant target = directTarget;
        if (target == null) {
            target = ui.chooseTarget(enemies);
        }

        // Invalid or dead target → no action
        if (target == null || !target.isAlive()) {
            return;
        }

        if (target.isImmune()) {
            ui.printInvulnerable(target);
            return;
        }

        int damage = Math.max(0, actor.getAttack() - target.getEffectiveDefense());

        target.takeDamage(damage);

        ui.printAttack(actor, target, damage);
    }

    @Override
    public String getName() {
        return "BasicAttack";
    }
}