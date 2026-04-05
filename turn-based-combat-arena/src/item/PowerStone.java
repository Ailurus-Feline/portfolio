package item;

import action.SpecialSkillAction;
import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

public class PowerStone implements Item {
    @Override
    public void use(Player player, List<Combatant> enemies, GameUI ui) {
        SpecialSkillAction skillAction = new SpecialSkillAction();
        int curCooldown = player.getCooldown();
        skillAction.execute(player, null, player, enemies, ui);
        if (player.getCooldown() != curCooldown) {
            restoreCooldown(player, curCooldown);
        }
        ui.printPowerStoneUsed(player);
    }

    private void restoreCooldown(Player player, int curCooldown) {
        try {
            java.lang.reflect.Field field = Player.class.getDeclaredField("cooldown");
            field.setAccessible(true);
            field.setInt(player, curCooldown);
        } catch (Exception ignored) {
        }
    }

    @Override
    public String getName() {
        return "Power Stone";
    }
}
