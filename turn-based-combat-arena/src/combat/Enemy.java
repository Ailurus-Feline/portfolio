package combat;

public abstract class Enemy extends Combatant {
    protected Enemy(String name, int maxHp, int attack, int defense, int speed) {
        super(name, maxHp, attack, defense, speed);
    }
}
