package action;

import combat.Combatant;
import combat.Player;
import combat.Warrior;
import combat.Wizard;
import effect.StunEffect;
import ui.GameUI;

import java.util.List;

public class SpecialSkillAction implements Action {
    @Override
    public void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui) {
        if (!(actor instanceof Player)) {
            return;
        }

        if (!player.canUseSkill()) {
            ui.printSkillOnCooldown(player);
            return;
        }

        if (player instanceof Warrior) {
            Combatant target = ui.chooseTarget(enemies);
            if (target == null) {
                return;
            }
            int damage = Math.max(0, player.getAttack() - target.getEffectiveDefense());
            target.takeDamage(damage);
            target.addStatusEffect(new StunEffect(2));
            ui.printShieldBash(player, target, damage);
            player.startCooldown();
            return;
        }

        if (player instanceof Wizard) {
            ui.printArcaneBlast(player);
            for (Combatant enemy : enemies) {
                if (!enemy.isAlive()) {
                    continue;
                }
                int damage = Math.max(0, player.getAttack() - enemy.getEffectiveDefense());
                enemy.takeDamage(damage);
                ui.printAttack(player, enemy, damage);
                if (!enemy.isAlive()) {
                    player.increaseAttack(10);
                    ui.printWizardAttackBoost(player);
                }
            }
            player.startCooldown();
        }
    }

    @Override
    public String getName() {
        return "SpecialSkill";
    }
}
