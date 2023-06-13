from src.prob_model import LitBot
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API active"}

@app.get("/initiate_bot/{game_id} {player_id} {player_count}")
async def initiate_bot(game_id : str, player_id : int, player_count : int):
    global bot
    bot = LitBot(game_id, player_id, 2, player_count, None, 0)
    return None
    

@app.get("/return_truth_matrix")
async def return_truth_matrix():
    return str(bot.truth_matrix)

@app.get("/test_func")
async def test_func():
    return "test_valid"