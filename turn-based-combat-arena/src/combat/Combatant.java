package combat;

import effect.StatusEffect;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public abstract class Combatant {
    private final String name;
    private final int maxHp;
    private int hp;
    private int attack;
    private int baseDefense;
    private int speed;
    private final List<StatusEffect> statusEffects;

    protected Combatant(String name, int maxHp, int attack, int baseDefense, int speed) {
        this.name = name;
        this.maxHp = maxHp;
        this.hp = maxHp;
        this.attack = attack;
        this.baseDefense = baseDefense;
        this.speed = speed;
        this.statusEffects = new ArrayList<>();
    }

    public void turnStart() {
        Iterator<StatusEffect> iterator = statusEffects.iterator();
        while (iterator.hasNext()) {
            StatusEffect effect = iterator.next();
            effect.turnStart(this);
            effect.decreaseDuration();
            if (effect.isExpired()) {
                effect.expire(this);
                iterator.remove();
            }
        }
    }

    public void turnEnd() {
    }

    public void addStatusEffect(StatusEffect effect) {
        statusEffects.add(effect);
    }

    public boolean isFrozen() {
        for (StatusEffect effect : statusEffects) {
            if (effect.freeze()) {
                return true;
            }
        }
        return false;
    }

    public boolean isImmune() {
        for (StatusEffect effect : statusEffects) {
            if (effect.immune()) {
                return true;
            }
        }
        return false;
    }

    public int getEffectiveDefense() {
        int total = baseDefense;
        for (StatusEffect effect : statusEffects) {
            total += effect.addDefense();
        }
        return total;
    }

    public void takeDamage(int damage) {
        hp = Math.max(0, hp - damage);
    }

    public void heal(int amount) {
        hp = Math.min(maxHp, hp + amount);
    }

    public boolean isAlive() {
        return hp > 0;
    }

    public String getName() {
        return name;
    }

    public int getMaxHp() {
        return maxHp;
    }

    public int getHp() {
        return hp;
    }

    public int getAttack() {
        return attack;
    }

    public void increaseAttack(int amount) {
        attack += amount;
    }

    public int getSpeed() {
        return speed;
    }

    public List<StatusEffect> getStatusEffects() {
        return new ArrayList<>(statusEffects);
    }
}
