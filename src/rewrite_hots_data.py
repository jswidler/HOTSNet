import numpy as np
import pandas as pd
import re
import os
import csv
from tqdm import tqdm

import hots_data as hots

MIN_LEVEL=1
MIN_MMR=700
MAX_MMR=4500

exportfile = os.path.join('data', 'hots_training_data.csv')

if os.path.isfile(exportfile):
    print("{} already exists.  Delete it before running this script if you want to create a new one.".format(exportfile))
    import sys
    sys.exit(1)


# The next few functions are responsible for taking the input data and creating a single
# vector per game.

def map_vector(map_id, map_vec, offset):
    '''
    Each map is assigned to a different map column, one of these is 1 while the rest are 0.
    '''
    map_vec[map_id-1001 + offset] = 1

def mode_vector(gamemode, mode_vec, offset):
    '''
    Each gamemode is assigned to a different mode column, one of these is 1 while the rest are 0.
    '''
    mode_vec[gamemode-hots.QUICK_MATCH + offset] = 1

def map_vector_labels():
    '''
    Create the column names for the map attributes in the game feature vector
    '''
    return ["map_" + hots.maps.iloc[i].Name for i in range(0,hots.maps.shape[0])]

def mode_vector_labels():
    '''
    Create the column names for the mode attributes in the game feature vector
    '''
    return ["mode_" + hots.gamemodes[i] for i in hots.gamemodes.keys()]

def by_mmr(i):
    '''
    Used to sort players by their mmr in team_vector function
    '''
    return -i[1]

def team_vector(team, team_vec, offset):
    '''
    Produce a vector for a team, or raise a ValueError if the team does not
    meet certain validation criteria.
    '''
    if len(team) != 5:
        raise ValueError
    for i, (hero_id, mmr, lvl) in enumerate(sorted(team[["HeroID", "MMR Before", "Hero Level"]].values, key=by_mmr)):
        if (hero_id < 1 or mmr < MIN_MMR or lvl < MIN_LEVEL or mmr > MAX_MMR or hero_id > 71
            or mmr != mmr or lvl != lvl or hero_id != hero_id):
            raise ValueError
        player_col = i * 2
        hero_col = 10 + int(hero_id - 1) * 2
        mmr_scaled = mmr/MAX_MMR
        level_scaled = (lvl + 5) / 25
        team_vec[player_col+offset] = mmr_scaled
        team_vec[hero_col+offset] = mmr_scaled
        team_vec[player_col+offset + 1] = level_scaled
        team_vec[hero_col+offset + 1] = level_scaled
        subgroup_col = 10 + hero_count * 2 + hots.hero_subgroups.index(hots.heroes.loc[hero_id].SubGroup)
        team_vec[subgroup_col+offset] += .33

def team_vector_labels(team='a'):
    '''
    Create the column names for the team attributes in the game feature vector
    '''
    labels = list()
    for i in range(1,6):
        labels.append(team+'_playermmr_' + str(i))
        labels.append(team+'_playerlevel_' + str(i))
    for i in range(1,len(hots.heroes) + 1):
        labels.append(team + '_herommr_' + hots.heroes.loc[i].Name)
        labels.append(team + '_herolevel_' + hots.heroes.loc[i].Name)
    for subgroup in hots.hero_subgroups:
        labels.append(team + '_subgroup_' + subgroup)
    return labels

def game_vector(replay_id):
    '''
    Produce a vector for a single game, or raise a ValueError if the data does not
    pass certain validation criteria.
    '''
    game_vec = [0.0] * v_len

    replay = hots.replays.loc[replay_id]
    winners = games.get_group((replay_id, 1))
    losers = games.get_group((replay_id, 0))

    mode_vector(replay.gamemode, game_vec, 1)
    map_vector(replay.map_id, game_vec, 5)

    # Randomly assign a team as 'a' and 'b'
    if np.random.randint(0, 2) == 0:
        game_vec[0] = 1
        team_vector(winners, game_vec, 5 + m_len)
        team_vector(losers, game_vec, 5 + m_len + t_len)
        return game_vec
    else:
        # game_vec[0] = 0
        team_vector(losers, game_vec, 5 + m_len)
        team_vector(winners, game_vec, 5 + m_len + t_len)
        return game_vec

def game_vector_labels():
    '''
    Create the column names for the game feature vector
    '''
    labels = ['team_a_won'] + mode_vector_labels() + map_vector_labels() + \
            team_vector_labels('a') + team_vector_labels('b')
    return list(map(lambda x: re.sub("[^a-zA-Z0-9_]","", x) , labels)) # Filter out unusual characters


# Below we will do the steps needed to produce a csv with our training data.

print("Loading HOTSLogs data...")
players = hots.load_replay_chars('all')
# We could also load smaller parts of the initial dataset by providing a game mode
# players = hots.load_replay_chars(hots.TEAM_LEAGUE)

replay_ids = list(players['ReplayID'].unique())
replay_count = len(replay_ids)
print("Number of games loaded: {}".format(replay_count))

# All of these variables are required to run the game_vector function
games = players.groupby(['ReplayID', 'Is Winner'])
hero_count=len(hots.heroes)
v_len = len(game_vector_labels())
m_len = len(map_vector_labels())
t_len = len(team_vector_labels())

# Print out some example data
while True:
    try:
        trow = pd.DataFrame(game_vector(np.random.choice(replay_ids)), index = game_vector_labels())
        break
    except ValueError:
        pass

print("\nSelected columns (out of {}) from a random game in the dataset:".format(v_len))
print(trow[:10])
print(trow[(v_len//2): (v_len//2)+29])

# Create 
print("\nRewriting game data to {}.".format(exportfile))
output_count = 0
with (open(exportfile,'w')) as file:
    writer=csv.writer(file)
    writer.writerow(game_vector_labels())
    for r_id in tqdm(replay_ids, unit="Games", mininterval=1):
        try:
            writer.writerow(game_vector(r_id))
            output_count += 1
        except ValueError:
            pass

print("\nComplete.  {} out of {} games in final dataset".format(output_count, replay_count))
