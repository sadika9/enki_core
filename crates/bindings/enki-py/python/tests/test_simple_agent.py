import enki_py.simple_agent as simple_agent_module


class FakeEnkiAgent:
    last_kwargs = None

    def __init__(self, handler):
        self.handler = handler

    @classmethod
    def with_tools(cls, **kwargs):
        cls.last_kwargs = kwargs
        return cls(kwargs["handler"])

    async def run(self, session_id: str, user_message: str) -> str:
        tool_names = {tool.name for tool in self.last_kwargs["tools"]}
        if {"roll_dice", "get_player_name"}.issubset(tool_names):
            guess = "".join(ch for ch in user_message if ch.isdigit())
            player_name = self.handler.execute("get_player_name", "{}", "", "", "")
            dice_roll = self.handler.execute("roll_dice", "{}", "", "", "")
            if dice_roll == guess:
                return f"Congratulations {player_name}, you guessed correctly! You're a winner!"
            return f"Sorry {player_name}, you guessed {guess} but rolled {dice_roll}."
        return "No-op"


def test_simple_agent_uses_only_external_tools(monkeypatch):
    monkeypatch.setattr(simple_agent_module, "_LowLevelEnkiAgent", FakeEnkiAgent)

    def roll_dice() -> str:
        """Roll a six-sided die and return the result."""
        return "4"

    def get_player_name(ctx: simple_agent_module.RunContext[str]) -> str:
        """Get the player's name."""
        return ctx.deps

    agent = simple_agent_module.SimpleAgent(
        "gateway/gemini:gemini-3-flash-preview",
        instructions=(
            "You're a dice game, you should roll the die and see if the number "
            "you get back matches the user's guess. If so, tell them they're a winner. "
            "Use the player's name in the response."
        ),
        deps_type=str,
        tools=[
            simple_agent_module.plain_tool(roll_dice),
            simple_agent_module.context_tool(get_player_name),
        ],
    )

    result = agent.run_sync("My guess is 4", deps="Anne")

    assert result.output == (
        "Congratulations Anne, you guessed correctly! You're a winner!"
    )
    assert FakeEnkiAgent.last_kwargs["include_builtin_tools"] is False
    assert [tool.name for tool in FakeEnkiAgent.last_kwargs["tools"]] == [
        "roll_dice",
        "get_player_name",
    ]
