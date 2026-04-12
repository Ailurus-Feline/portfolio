package action;

import combat.Combatant;
import combat.Player;
import combat.Warrior;
import combat.Wizard;
import effect.StunEffect;
import ui.GameUI;

import java.util.List;

/**
 * Executes class-specific special skill for a player.
 *
 * Warrior: single-target damage + stun (2 turns).
 * Wizard: AOE + permanent attack boost on kill.
 * Skill is subject to cooldown.
 */
public class SpecialSkillAction implements CooldownAction {

    @Override
    public void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui, boolean ifCooldown) {

        if (!(actor instanceof Player)) {
            return;
        }

        if (ifCooldown && !player.canUseSkill()) {
            ui.printSkillOnCooldown(player);
            return;
        }

        if (player instanceof Warrior) {
            Combatant target = directTarget;
            
            if (target == null) {
                target = ui.chooseTarget(enemies);
            }

            if (target == null || !target.isAlive()) {
                return;
            }

            int damage = Math.max(0, player.getAttack() - target.getEffectiveDefense());
            target.takeDamage(damage);

            // Apply stun effect (2 turns)
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

                // Gain attack if enemy is defeated
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