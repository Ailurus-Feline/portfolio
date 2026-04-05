package effect;

import combat.Combatant;

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

    public boolean freeze() {
        return false;
    }

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
