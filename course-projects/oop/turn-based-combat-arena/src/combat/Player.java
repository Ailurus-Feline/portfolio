package combat;

import item.Item;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for player-controlled characters.
 *
 * Extends Combatant with inventory management and skill cooldown.
 */
public abstract class Player extends Combatant {

    private final List<Item> inventory;
    private int cooldown;

    protected Player(String name, int maxHp, int attack, int defense, int speed) {
        super(name, maxHp, attack, defense, speed);
        this.inventory = new ArrayList<>();
        this.cooldown = 0;
    }

    /**
     * Decreases skill cooldown at end of turn.
     */
    @Override
    public void turnEnd() {
        super.turnEnd();
        if (cooldown > 0) {
            cooldown--;
        }
    }

    public void addItem(Item item) {
        inventory.add(item);
    }

    public List<Item> getInventory() {
        return inventory;
    }

    public boolean canUseSkill() {
        return cooldown == 0;
    }

    public void startCooldown() {
        cooldown = 3;
    }

    public int getCooldown() {
        return cooldown;
    }
}