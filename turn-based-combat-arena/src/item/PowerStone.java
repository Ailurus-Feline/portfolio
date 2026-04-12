package item;

import action.SpecialSkillAction;
import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

/**
 * Item that triggers player's special skill without consuming cooldown.
 */
public class PowerStone implements Item {

    @Override
    public void use(Player player, List<Combatant> enemies, GameUI ui) {
        SpecialSkillAction skillAction = new SpecialSkillAction();

        skillAction.execute(player, null, player, enemies, ui, false);

        ui.printPowerStoneUsed(player);
    }

    @Override
    public String getName() {
        return "Power Stone";
    }
}