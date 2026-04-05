package action;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

public class BasicAttack implements Action {
    @Override
    public void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui) {
        Combatant target = directTarget;
        if (target == null) {
            target = ui.chooseTarget(enemies);
        }

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
