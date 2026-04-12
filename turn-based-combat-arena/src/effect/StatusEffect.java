package effect;

import combat.Combatant;

/**
 * Base class for all status effects.
 *
 * Defines basic methods for status effect behavior.
 */
public abstract class StatusEffect {

    private final String name;
    private int duration;

    protected StatusEffect(String name, int duration) {
        this.name = name;
        this.duration = duration;
    }

    public void turnStart(Combatant target) {
    }

    public void expire(Combatant target) {
    }

    /**
     * returns true if this effect prevents the target from acting
     */
    public boolean freeze() {
        return false;
    }

    /**
     * returns true if this effect has damage immunity
     */
    public boolean immune() {
        return false;
    }

    public int addDefense() {
        return 0;
    }

    public void decreaseDuration() {
        duration--;
    }

    public boolean isExpired() {
        return duration <= 0;
    }

    public String getName() {
        return name;
    }

    public int getDuration() {
        return duration;
    }
}