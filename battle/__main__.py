from battle.environment.environment import Environment


def main() -> None:
    env = Environment()
    cities = env.cities
    env.step([[(cities[1], 2000)]], {})
    env.step([], {0: cities[0]})
    print(
        f"""
        cities: {env.cities},
        armies: {env.armies}
    """
    )


if __name__ == "__main__":
    main()
