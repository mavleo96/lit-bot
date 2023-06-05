import pandas as pd
import numpy as np
import itertools
import json
import copy

class LitBot:
    EPSILON = 0.01
    def __init__(self, game_id, team_id, player_id, player_count, game_state_array=None, fake_game_seed = None):
        self.game_id = game_id
        self.team_id = team_id
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
            self.initialize_matrix(self.team_id, self.player_id, self._fake_initial_game_state_array[0, 0])
        elif game_state_array is not None:
            self.initialize_matrix(self.team_id, self.player_id, game_state_array)
   
    def initialize_fake_game(self, fake_game_seed):
        
        np.random.seed(fake_game_seed)
        card_location_array = np.random.choice(range(self.player_count), (8,6))
        card_location_array_team = card_location_array%2
        card_location_array_player = card_location_array//2

        game_state_array = np.zeros((2, self.player_count//2, 8, 6))
        for set_id, card_id in itertools.product(range(8), range(6)):
            game_state_array[
                card_location_array_team[set_id, card_id],
                card_location_array_player[set_id, card_id],
                set_id,        
                card_id
            ] = 1
        
        return game_state_array, card_location_array

    def initialize_matrix(self, team_id, player_id, player_array):
        
        self.truth_matrix = np.full((2, self.player_count//2, 8, 6), LitBot.EPSILON)
        self.truth_matrix[team_id, player_id, :, :] = player_array
        self.truth_matrix[team_id, [i for i in np.arange(self.player_count//2) if i != player_id ], :, :] -= player_array*LitBot.EPSILON
        self.truth_matrix[0 if team_id == 1 else 1, [i for i in np.arange(self.player_count//2)], :, :] -= player_array*LitBot.EPSILON
        
        self.inference_matrix = copy.deepcopy(self.truth_matrix)

        self.active_cards_matrix = np.zeros((8, 6))
        self.active_sets_matrix = np.zeros(8)
        self.recent_card_array = np.full((8, 2), -1)

        self.player_card_counter = np.full((2, self.player_count//2), 48//self.player_count)
        self.player_set_card_counter = np.full((2, self.player_count//2, 8), LitBot.EPSILON)

        self.prob_matrix = np.zeros((0, 2, self.player_count//2, 8, 6))
        self.prob_matrix = self.update_prob_matrix(self.truth_matrix)

        # self.total_shannon_info = -np.log2(np.full((8,6), 1/self.player_count))
        # self.shannon_info_matrix = self.update_shannon_info_matrix(self.prob_matrix)
        
        
    def update_prob_matrix(self, truth_matrix):

        prob_matrix = np.zeros((1, 2, self.player_count//2, 8, 6))
        prob_matrix[0, :, :, :, :] += truth_matrix

        for set_id, card_id in itertools.product(range(0,8), range(0,6)):
            prob_matrix[0, :, :, set_id, card_id] = np.where(
                truth_matrix[:, :, set_id, card_id] == LitBot.EPSILON,
                LitBot.EPSILON/truth_matrix[:, :, set_id, card_id].sum(),
                truth_matrix[:, :, set_id, card_id]
            )
                
        # self.prob_matrix = prob_matrix
        return prob_matrix

    def update_shannon_info_matrix(self, prob_matrix):

        uncertainity_matrix = prob_matrix.max(axis=(0, 1), keepdims=True)
        shannon_info_matrix = self.total_shannon_info + np.log2(uncertainity_matrix)

        # self.shannon_info_matrix = shannon_info_matrix
        return shannon_info_matrix

    def update_active_sets(self, set_id, card_id, result):
        self.active_sets_matrix[set_id] = 1
        if result == 1:
            print("removed active card")
            self.active_cards_matrix[set_id, card_id] = 0
        else:
            print("added active card")
            self.active_cards_matrix[set_id, card_id] = 1

    def update_inference_matrix(self, ask_player_id, ask_team_id, set_id, card_id):
        # if self.active_cards_matrix[set_id].sum() > 0:
        #     if self.active_cards_matrix[set_id, card_id] == 0:
        #         self.inference_matrix[
        #             ask_player_id,
        #             set_id,
        #             np.where(self.active_cards_matrix[set_id] == 1)[0].tolist()
        #         ] = 1/self.active_cards_matrix[set_id].sum()
        
        if (self.recent_card_array[set_id, 0] != card_id) and (self.recent_card_array[set_id, 1] == ask_team_id) and (self.active_cards_matrix[set_id, self.recent_card_array[set_id, 0]] == 1):
            print("updated deep inference matrix")
            self.inference_matrix[
                ask_team_id,
                ask_player_id,
                set_id,
                self.recent_card_array[set_id, 0]
            ] = 1

    def update_player_card_count(self, ask_team_id, ask_player_id, ans_team_id, ans_player_id, set_id, card_id, result):

        if self.player_set_card_counter[ask_team_id, ask_player_id, set_id] == LitBot.EPSILON:
            print("new player set combination detected")
            self.player_set_card_counter[ask_team_id, ask_player_id, set_id] = 1

        if result == 1:
            print("updated card count")
            self.player_card_counter[ask_team_id, ask_player_id] += 1
            self.player_card_counter[ans_team_id, ans_player_id] -= 1

            print("updated player set card count")
            self.player_set_card_counter[ask_team_id, ask_player_id, set_id] += 1

        self.truth_matrix_completeness_inference()
        self.check_count_completeness_inference()

    def truth_matrix_completeness_inference(self):

        for set_id, card_id in itertools.product(range(0,8), range(0,6)):
            if self.inference_matrix[:, :, set_id, card_id].sum() == LitBot.EPSILON:
                print("updating truth matrices on completeness logic")
                team_id = np.argwhere(self.inference_matrix[:, :, set_id, card_id] == LitBot.EPSILON)[0][0]
                player_id = np.argwhere(self.inference_matrix[:, :, set_id, card_id] == LitBot.EPSILON)[0][1]
                self.inference_matrix[team_id, player_id, set_id, card_id] = 1
                self.truth_matrix[team_id, player_id, set_id, card_id] = 1

    def check_count_completeness_inference(self):
        
        player_set_card_count = np.where(self.player_set_card_counter == LitBot.EPSILON, 0, self.player_set_card_counter)

        # set completeness
        set_card_count = player_set_card_count.sum(axis=(0, 1))
        completed_sets = np.argwhere(set_card_count==6).reshape(-1).tolist()


        for set_id in completed_sets:
            print(f"set_id {set_id} is semi certain now")
            null_players = np.argwhere(player_set_card_count[:, :, set_id] == 0).tolist()
            for team_id, player_id in null_players:
                print(f"team_id {team_id} player_id {player_id} doesn't have set_id {set_id}")
                self.truth_matrix[team_id, player_id, set_id, :] = 0
                # np.where(self.truth_matrix[team_id, player_id, set_id, :] == LitBot.EPSILON, 0, self.truth_matrix[:, :, set_id, :])
                self.inference_matrix[team_id, player_id, set_id, :] = 0
                # np.where(self.inference_matrix[:, :, set_id, :] == LitBot.EPSILON, 0, self.inference_matrix[:, :, set_id, :])

        # player card completeness
        player_card_count = player_set_card_count.sum(axis=(2))
        completed_players = np.argwhere(player_card_count==self.player_card_counter).tolist()
        
        for team_id, player_id in completed_players:
            print(f"team_id {team_id} player_id {player_id}  is semi certain now")
            null_sets = np.argwhere(player_set_card_count[team_id, player_id, :] == 0).tolist()
            for set_id in null_sets:
                print(f"team_id {team_id} player_id {player_id} doesn't have set_id {set_id}")
                self.truth_matrix[team_id, player_id, set_id, :] = 0
                # np.where(self.truth_matrix[team_id, player_id, :, :] == LitBot.EPSILON, 0, self.truth_matrix[team_id, player_id, :, :])
                self.inference_matrix[team_id, player_id, set_id, :] = 0
                # np.where(self.inference_matrix[team_id, player_id, :, :] == LitBot.EPSILON, 0, self.inference_matrix[team_id, player_id, :, :])

    def update_game(self, game_action_dict, stop = True):
        
        if game_action_dict["action"] == "ask_card":

            ask_team_id = game_action_dict["by_team"]
            ans_team_id = game_action_dict["to_team"]
            ask_player_id = game_action_dict["by"]
            ans_player_id = game_action_dict["to"]
            set_id = game_action_dict["set_id"]
            card_id = game_action_dict["card_id"]
            result = game_action_dict["result"]

            if result == 1:
                print("updated truth matrix 1")
                self.truth_matrix[ask_team_id, ask_player_id, set_id, card_id] = 1
                self.truth_matrix[ask_team_id, [i for i in np.arange(self.player_count//2) if i != ask_player_id ], set_id, card_id] = 0
                self.truth_matrix[ans_team_id, [i for i in np.arange(self.player_count//2)], set_id, card_id] = 0

                print("updated inference matrix 1")
                self.inference_matrix[ask_team_id, ask_player_id, set_id, card_id] = 1
                self.inference_matrix[ask_team_id, [i for i in np.arange(self.player_count//2) if i != ask_player_id ], set_id, card_id] = 0
                self.inference_matrix[ans_team_id, [i for i in np.arange(self.player_count//2)], set_id, card_id] = 0


            else:
                print("updated truth matrix 0")
                self.truth_matrix[ask_team_id, ask_player_id, set_id, card_id] = 0
                self.truth_matrix[ans_team_id, ans_player_id, set_id, card_id] = 0
                print("updated inference matrix 0")
                self.inference_matrix[ask_team_id, ask_player_id, set_id, card_id] = 0
                self.inference_matrix[ans_team_id, ans_player_id, set_id, card_id] = 0
                
                self.update_inference_matrix(ask_player_id, ask_team_id, set_id, card_id)

            self.update_player_card_count(ask_team_id, ask_player_id, ans_team_id, ans_player_id, set_id, card_id, result)


            self.update_active_sets(set_id, card_id, result)
            self.recent_card_array[set_id, 0] = card_id
            self.recent_card_array[set_id, 1] = ask_team_id
            self.prob_matrix = self.update_prob_matrix(self.truth_matrix)
            # self.shannon_info_matrix = self.update_shannon_info_matrix(self.prob_matrix)

# truth_matrix (player_id, set_id, card_id)
# prob_matrix  (0, player_id, set_id, card_id)