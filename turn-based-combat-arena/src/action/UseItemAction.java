package action;

import combat.Combatant;
import combat.Player;
import item.Item;
import ui.GameUI;

import java.util.List;

public class UseItemAction implements Action {
    @Override
    public void execute(Combatant actor, Combatant directTarget, Player player, List<Combatant> enemies, GameUI ui) {
        if (!(actor instanceof Player)) {
            return;
        }

        Item item = ui.chooseItem(player);
        if (item == null) {
            ui.printNoItemUsed();
            return;
        }

        item.use(player, enemies, ui);
        player.getInventory().remove(item);
    }

    @Override
    public String getName() {
        return "Item";
    }
}
