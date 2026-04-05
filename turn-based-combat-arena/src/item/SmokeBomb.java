package item;

import combat.Combatant;
import combat.Player;
import effect.InvulnerabilityEffect;
import ui.GameUI;

import java.util.List;

public class SmokeBomb implements Item {
    @Override
    public void use(Player player, List<Combatant> enemies, GameUI ui) {
        player.addStatusEffect(new InvulnerabilityEffect(2));
        ui.printSmokeBombUsed(player);
    }

    @Override
    public String getName() {
        return "Smoke Bomb";
    }
}
