import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class LiteratureGameAbstract(ABC):
    def __init__(
            self, 
            team_count : int,
            player_count : int,
            set_count : int = 8,
            set_card_count : int = 6,
        ) -> None:
        
        self.set_count = set_count
        self.set_card_count = set_card_count
        
        self.team_count = team_count
        self.player_count = player_count
        
        self.check_game_inputs()
        
    def check_game_inputs(self) -> None:

        inputs = {
            "team_count" : self.team_count, 
            "player_count" : self.player_count, 
            "set_count" : self.set_count, 
            "set_card_count" : self.set_card_count
        }

        issue_inputs = [key for key, value in inputs.items() if isinstance(value, int) == False]
        if len(issue_inputs) > 0:
            raise ValueError(f"Inputs must be integers: {issue_inputs}")
        
        if self.player_count%self.team_count != 0:
            raise ValueError(f"player_count must be divisible by team_count: {self.player_count} % {self.team_count} != 0")
        if self.total_card_count%self.player_count != 0:
            raise ValueError(f"total_card_count must be divisible by player_count: {self.total_card_count} % {self.player_count} != 0")

    @property
    def total_card_count(self) -> int:
        return self.set_count*self.set_card_count

    @property
    def player_per_team_count(self) -> int:
        return self.player_count//self.team_count

    @property
    def card_per_player_count(self) -> int:
        return self.total_card_count//self.player_count
    
class LiteratureBotAbstract(LiteratureGameAbstract):
    def __init__(
            self, 
            team_id : int,
            player_id : int,
            team_count : int,
            player_count : int,
            initial_game_state_array : np.ndarray = None,
            set_count : int = 8,
            set_card_count : int = 6,
            fake_game_seed : int = None
        ) -> None:
        
        super().__init__(
            team_count = team_count,
            player_count = player_count,
            set_count = set_count,
            set_card_count = set_card_count
        )

        self.team_id = team_id
        self.player_id = player_id

        self.initial_game_state_array = initial_game_state_array
        self.fake_game_seed = fake_game_seed
        
        self.check_bot_inputs()

        if self.fake_game_seed is not None:
            (
                self._fake_initial_game_state_array,
                self._fake_initial_card_location_array
            ) = self.initialize_fake_game()
        elif self.initial_game_state_array is None:
            raise ValueError("Must provide either game_state_array or fake_game_seed")
        
    def check_bot_inputs(self) -> None:

        inputs = {
            "team_id" : self.team_id, 
            "player_id" : self.player_id,
        }

        issue_inputs = [key for key, value in inputs.items() if isinstance(value, int) == False]
        if len(issue_inputs) > 0:
            raise ValueError(f"Inputs must be integers: {issue_inputs}")
        if self.initial_game_state_array is not None and isinstance(self.initial_game_state_array, np.ndarray) == False:
            raise ValueError(f"initial_game_state_array must be a numpy.ndarray: {type(self.initial_game_state_array)}")
        if self.fake_game_seed is not None and isinstance(self.fake_game_seed, int) == False:
            raise ValueError(f"fake_game_seed must be an integer: {type(self.fake_game_seed)}")
        
        if self.team_id >= self.team_count:
            raise ValueError(f"team_id must be less than team_count: {self.team_id} >= {self.team_count}")
        if self.player_id >= self.player_per_team_count:
            raise ValueError(f"player_id must be less than player_count_per_team: {self.player_id} >= {self.player_per_team_count}")

    def initialize_fake_game(self) -> Tuple[np.ndarray, np.ndarray]:
        dummy_card_location_array = np.tile(np.arange(self.player_count), self.card_per_player_count).reshape(-1)
        np.random.seed(self.fake_game_seed)
        np.random.shuffle(dummy_card_location_array)
        card_location_array = dummy_card_location_array.reshape(self.set_count, self.set_card_count)
        
        game_state_array = np.array(
            [np.where(card_location_array == i, 1, 0).tolist() for i in range(self.player_count)]
        ).reshape(self.team_count, self.player_per_team_count, self.set_count, self.set_card_count)

        return game_state_array, card_location_array

    @abstractmethod
    def initialize_matrix(self) -> Tuple:
        pass

    @abstractmethod
    def update_game(self, game_action_dict, stop = True) -> None:
        pass