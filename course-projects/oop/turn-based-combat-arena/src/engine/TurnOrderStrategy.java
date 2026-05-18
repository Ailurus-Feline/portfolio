package engine;

import combat.Combatant;

import java.util.List;

public interface TurnOrderStrategy {
    List<Combatant> whatOrder(List<Combatant> combatants);
}
