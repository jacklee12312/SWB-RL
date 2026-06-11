from swb.engine.abilities import ABILITY_DEFINITIONS


def main() -> None:
    print(f"{'能力':<10} {'状态':<12} 触发事件")
    print("-" * 60)
    for definition in ABILITY_DEFINITIONS:
        events = ", ".join(event.value for event in definition.events) or "static"
        print(f"{definition.keyword.value:<10} {definition.status.value:<12} {events}")


if __name__ == "__main__":
    main()
