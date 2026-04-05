import engine.BattleEngine;
import ui.GameUI;

public class Main {
    public static void main(String[] args) {
        GameUI ui = new GameUI();
        BattleEngine engine = new BattleEngine(ui);
        engine.startGame();
    }
}
