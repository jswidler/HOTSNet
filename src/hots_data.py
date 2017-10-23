from tqdm import tqdm
import requests
import os
import zipfile
import pandas as pd

QUICK_MATCH=3
HERO_LEAGUE=4
TEAM_LEAGUE=5
UNRANKED_DRAFT=6
gamemodes= {
    QUICK_MATCH: 'Quick Match',
    HERO_LEAGUE: 'Hero League',
    TEAM_LEAGUE: 'Team League', 
    UNRANKED_DRAFT: 'Unranked Draft'
}

zip_path = os.path.join("data", "HOTSLogs_Data.zip")
replay_path = os.path.join("data", "Replays.csv")
rchars_path = os.path.join("data", "ReplayCharacters.csv")
heromap_path = os.path.join("data", "HeroIDAndMapID.csv")

def check_for_files():
    return os.path.isfile(replay_path) and os.path.isfile(rchars_path) and os.path.isfile(heromap_path)

def download_hots_data():
    if os.path.isfile(zip_path):
        print('{} exists already'.format(zip_path))
        return 
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print('Downloading HOTSLogs rolling 30 day game export:')
    r = requests.get("https://d1i1jxrdh2kvwy.cloudfront.net/Data/HOTSLogs%20Data%20Export%20Current.zip", stream=True)
    with open(zip_path, "wb") as f:
        for chunk in tqdm(
            r.iter_content(32*1024),
            total=int(int(r.headers['content-length'])/(32*1024)),
            unit=' 32kB chunks'
        ):
            f.write(chunk)
            
def extract_hots_data():
    print('Unzipping {}'.format(zip_path))
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall("data")
    zip_ref.close()
    
def get_files():       
    if check_for_files():
        print('HOTSLogs data found.')
    else:
        download_hots_data()
        extract_hots_data()
        os.remove(zip_path)
        if not check_for_files():
            raise RuntimeError('Failed to download or extract HOTSLogs Export.')

heroes = None
hero_groups = None
hero_subgroups = None
maps = None
def load_hero_map_data():
    global heroes, hero_groups, hero_subgroups, maps
    if maps is None:
        heromap = pd.read_csv(heromap_path, index_col = 0)
        heroes = heromap.loc[1:99]
        hero_groups = list(heroes.groupby('Group').groups.keys())
        hero_subgroups = list(heroes.groupby('SubGroup').groups.keys())
        maps = heromap.loc[999:]
        maps = maps.drop(['Group', 'SubGroup'], axis = 1)

replays = None
def load_replays():
    global replays
    load_hero_map_data()
    if replays is None:
        replays = pd.read_csv(replay_path)
        replays.index = replays["ReplayID"]
        replays.columns = ["replay_id", "gamemode", "map_id", "duration", "timestamp"]
    return replays

gametype = None
replay_chars = None
def load_replay_chars(type='all'):
    '''
    Load replay chars into panda dataframe
    By default this will load the whole file.  However, type can be set to 3-6 to load only games of a given type
    '''
    global gametype
    global replay_chars
    load_replays()
    if gametype == type:
        return replay_chars
    if type == 'all':
        replay_chars = pd.read_csv(rchars_path, header=0)
    else:
        columns = pd.read_csv(rchars_path, nrows = 0, skiprows=0, header=0)
        skip = replays[replays.gamemode < type].shape[0] * 10
        keep = replays[replays.gamemode == type].shape[0] * 10
        replay_chars = pd.read_csv(rchars_path, nrows = keep, skiprows=skip, header=0)
        replay_chars.columns = columns.columns
    gametype = type
    return replay_chars

        
get_files()