package item;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

public class Potion implements Item {
    @Override
    public void use(Player player, List<Combatant> enemies, GameUI ui) {
        int curHp = player.getHp();
        player.heal(100);
        ui.printPotionUsed(player, curHp, player.getHp());
    }

    @Override
    public String getName() {
        return "Potion";
    }
}
