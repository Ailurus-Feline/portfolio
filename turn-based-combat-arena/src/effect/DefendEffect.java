package effect;

public class DefendEffect extends StatusEffect {
    public DefendEffect(int duration) {
        super("Defend", duration);
    }

    @Override
    public int addDefense() {
        return 10;
    }
}
