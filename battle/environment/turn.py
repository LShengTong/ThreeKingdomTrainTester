from typing import List

import torch

from battle.environment.army import Army
from battle.environment.city import City
from battle.environment.config import Config
from battle.environment.i_attack import IAttack


class Turn:
    @staticmethod
    def exec(cities: List[City], armies: List[Army]) -> City | None:
        curr_time = 0
        attacks: List[IAttack] = armies + cities
        attack_time = 0
        while curr_time < Config.turn_interval:
            if attack_time >= Config.attack_interval:
                attack_time -= Config.attack_interval
                winner = Turn._do_attack(attacks)
                if winner is not None:
                    return winner
            for army in armies:
                if army.is_alive():
                    army.move_to_target(Config.turn_delta_time)
            curr_time += Config.turn_delta_time
            attack_time += Config.turn_delta_time
        return None

    @staticmethod
    def _do_attack(attacks: List[IAttack]) -> City | None:
        for attacker in attacks:
            if not attacker.is_alive():
                continue
            defenders: List[IAttack] = []
            for defender in attacks:
                if (attacker.get_city_id() == defender.get_city_id() or
                        not defender.is_alive()):
                    continue
                dist = torch.dist(attacker.get_pos(), defender.get_pos())
                if dist < Config.attack_radius:
                    defenders.append(defender)
            for defender in defenders:
                defender.defense(int(attacker.attack() / len(defenders)))
                if not defender.is_alive() and isinstance(defender, City):
                    return defender
        return None