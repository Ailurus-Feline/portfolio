package effect;

public class StunEffect extends StatusEffect {
    public StunEffect(int duration) {
        super("Stun", duration);
    }

    @Override
    public boolean freeze() {
        return true;
    }
}
