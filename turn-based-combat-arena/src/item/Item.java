package item;

import combat.Combatant;
import combat.Player;
import ui.GameUI;

import java.util.List;

public interface Item {
    void use(Player player, List<Combatant> enemies, GameUI ui);
    String getName();
}
