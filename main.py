import pandas as pd
import numpy as np
import itertools
import json

class LitBot:
    EPSILON = 0.001
    def __init__(self, game_id, player_id, player_count, fake_game_seed = None):
        self.game_id = game_id
        self.player_id = player_id
        self.player_count = player_count

        self.encoding = {
            "set_id" : {
                0 : "lower_hearts",
                1 : "lower_diamonds",
                2 : "lower_spades",
                3 : "lower_clubs",
                4 : "upper_hearts",
                5 : "upper_diamonds",
                6 : "upper_spades",
                7 : "upper_clubs",
            },
            "set_id_type" : {
                0 : "lower",
                1 : "lower",
                2 : "lower",
                3 : "lower",
                4 : "upper",
                5 : "upper",
                6 : "upper",
                7 : "upper",
            },
            "card_id" : {
                "lower" : {
                    0 : "ace",
                    1 : "two",
                    2 : "three",
                    3 : "four",
                    4 : "five",
                    5 : "six",
                },
                "upper" : {
                    0 : "eight",
                    1 : "nine",
                    2 : "ten",
                    3 : "jack",
                    4 : "queen",
                    5 : "king",
                }
            }
        }
        if fake_game_seed is not None:
            (
                self._fake_initial_game_state_array,
                self._fake_initial_card_location_array
            ) =  self.initialize_fake_game(fake_game_seed)
            self.initialize_matrix(self.player_id, self._fake_initial_game_state_array[0])
   
    def initialize_fake_game(self, fake_game_seed):
        
        np.random.seed(fake_game_seed)
        card_location_array = np.random.choice(range(self.player_count), (8,6))

        game_state_array = np.zeros((self.player_count, 8, 6))
        for set_id, card_id in itertools.product(range(8), range(6)):
            game_state_array[
                card_location_array[set_id, card_id],
                set_id,        
                card_id
            ] = 1
        
        return game_state_array, card_location_array

    def _get_random_generator_state(self):

        with open('randomiser_state.json', mode="r") as file:
            random_generator_state = tuple(json.loads(file.read()))

        self.random_generator_state = random_generator_state

    def initialize_matrix(self, player_id, player_array):
        
        self.truth_matrix = np.full((self.player_count, 8, 6), LitBot.EPSILON)
        self.truth_matrix[player_id, :, :] = player_array
        self.truth_matrix[[i for i in np.arange(self.player_count) if i != player_id ], :, :] -= player_array*LitBot.EPSILON
        
        self.prob_matrix = np.zeros((0, self.player_count, 8, 6))
        self.update_prob_matrix(self.truth_matrix)

        self.total_shannon_info = -np.log2(np.full((8,6), 1/self.player_count))
        self.update_shannon_info_matrix(self.prob_matrix)
        
        
    def update_prob_matrix(self, truth_matrix):

        prob_matrix = np.zeros((1, self.player_count, 8, 6))
        prob_matrix[0, :, :, :] += truth_matrix

        for set_id, card_id in itertools.product(range(0,8), range(0,6)):
            prob_matrix[0, :, set_id, card_id] = np.where(
                truth_matrix[:, set_id, card_id] == LitBot.EPSILON,
                LitBot.EPSILON/truth_matrix[:, set_id, card_id].sum(),
                truth_matrix[:, set_id, card_id]
            )
                
        self.prob_matrix = prob_matrix
    
    def update_shannon_info_matrix(self, prob_matrix):

        uncertainity_matrix = prob_matrix.max(axis=(0, 1), keepdims=True)

        self.shannon_info_matrix = self.total_shannon_info + np.log2(uncertainity_matrix)


    def update_game(self, game_action_dict):
        
        if game_action_dict["action"] == "ask_card":

            ask_player_id = game_action_dict["by"]
            ans_player_id = game_action_dict["to"]
            set_id = game_action_dict["set_id"]
            card_id = game_action_dict["card_id"]

            if game_action_dict["result"] == 1:
                self.truth_matrix[ask_player_id, set_id, card_id] = 1
                self.truth_matrix[[i for i in np.arange(self.player_count) if i != ask_player_id ], set_id, card_id] = 0
            if game_action_dict["result"] == 0:
                self.truth_matrix[[ask_player_id, ans_player_id], set_id, card_id] = 0

            self.update_prob_matrix(self.truth_matrix)
            self.update_shannon_info_matrix(self.truth_matrix)
    
from fastapi import FastAPI

app = FastAPI()
bot = LitBot(0, int(0), int(6))

@app.get("/initiate_bot/{game_id} {player_id} {player_count}")
async def initiate_bot(game_id : str, player_id : int, player_count : int):
    # global bot 
    return None
    

@app.get("/return_truth_matrix")
async def return_truth_matrix():
    return str(bot.truth_matrix)

@app.get("/test_func")
async def test_func():
    return "test_valid"