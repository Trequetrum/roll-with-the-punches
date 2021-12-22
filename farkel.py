from typing import List
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

def flipped_bits(n: int) -> List[int]:
    bits = []
    for i in range(6):
        if(n & 1 == 1): 
            bits.append(i)
        n = n >> 1
    return bits


def contained(candidate, container):
    """ Check if candidate can be constructed by only removing 
    elements from container. Instead of subset, this is sublist

    There's probably a better way to do this using clever sorting
    and indexing. I can find a library function that does this for 
    me.
    """
    temp = container[:]
    try:
        for v in candidate:
            temp.remove(v)
        return True
    except ValueError:
        return False

"""
    1 	100 points
    5 	50 points
    Three 1's 	1,000 points
    Three 2's 	200 points
    Three 3's 	300 points
    Three 4's 	400 points
    Three 5's 	500 points
    Three 6's 	600 points
    1-2-3-4-5-6 3000 points
    ### Not included:
        3 Pairs 	1500 points
"""
SCORING_FARKEL = {
    'SCORE_FARKEL': ([1,2,3,4,5,6], 3000),
    'SCORE_ONES': ([1,1,1], 1000),
    'SCORE_TWOS': ([2,2,2], 200),
    'SCORE_THREES': ([3,3,3], 300),
    'SCORE_FOURS': ([4,4,4], 400),
    'SCORE_FIVES': ([5,5,5], 500),
    'SCORE_SIXES': ([6,6,6], 600),
    'SCORE_ONE': ([1], 100),
    'SCORE_FIVE': ([5], 50)
}

def farkel_score(*die: int) -> int:

    if not die:
        return 0
    
    dice = list(die)
    score = 0

    for (pattern, points) in SCORING_FARKEL.values():
        while True:
            temp_dice = dice[:]
            try:
                for v in pattern:
                    temp_dice.remove(v)
                
                dice = temp_dice
                score += points
            except ValueError:
                break

    return score

    


class FarkelEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FarkelEnv, self).__init__()
        # ----- Action Space
        # An action for each way you can roll a die. So a 127 means the
        # action of rolling die 1,2,3,4,5,&,6. A 34 means the action of
        # rolling die 2 & 6. Think of this as a binary encoding of which
        # die are rolled.
        self.action_space = spaces.Discrete(127)

        # ----- Observation space
        # Banked score: 0 - 1,000,000 (discrete)
        # Betting score: 0 - 1,000,000 (discrete)
        # Die Configuration:
        #   Each die is 0 - 6 (discrete), represending the current value
        #   of that die. A 0 means the die can no longer be rolled or
        #   scored this round.
        self.observation_space = spaces.MultiDiscrete(
            [1000000, 1000000, 6, 6, 6, 6, 6, 6], dtype=np.int64)

        self.environment = {
            'banked_score': int(0),
            'betting_score': int(0),
            'dice': np.zeros(6, dtype=np.int64)
        }

        self.observable_environment = np.zeros(8, dtype=np.int64)
        self.reset()

    def get_observable_environment(self):
        self.observable_environment[0] = self.environment['banked_score']
        self.observable_environment[1] = self.environment['betting_score']
        for i in range(6):
            self.observable_environment[i+2] = self.environment['dice'][i]
        return self.observable_environment

    def step(self, action):

        print(f"action: {action}")

        rolling_dice = [x for x in flipped_bits(action)]

        #------ No-op actions ---------
        # You can't pass if you have less than 300 banked points
        if action == 0 and self.environment['banked_score'] < 300:
            return self.observable_environment, 0, False, {}
        # You can't roll a scored die
        for d in rolling_dice:
            if self.environment['dice'][d] == 0:
                print("You can't roll a scored die")
                return self.observable_environment, 0, False, {}
        #------------------------------
        
        # Re-roll the rolling dice
        for d in rolling_dice:
            self.environment['dice'][d] = np.random.randint(1,6)

        scoring_dice = [x for x in range(6) if x not in rolling_dice]

        scoring_dice_v = [
            self.environment['dice'][i] 
            for i in scoring_dice 
            if self.environment['dice'][i] != 0
        ]

        score = farkel_score(*scoring_dice_v)

        # A scored die is removed from the game for the round
        for d in scoring_dice:
            self.environment['dice'][d] = 0


        if score > 0 and action == 0:
            self.environment['banked_score'] += self.environment['betting_score']
            self.environment['betting_score'] = 0
            return self.get_observable_environment(), self.environment['banked_score'], True, {}
        elif score > 0:
            self.environment['betting_score'] += score
            return self.get_observable_environment(), self.environment['betting_score'], False, {}
        else:
            self.environment['betting_score'] = 0
            return self.get_observable_environment(), self.environment['banked_score'], True, {}

    def reset(self):

        print("reset")

        self.environment['banked_score'] = 0
        self.environment['betting_score'] = 0
        score = 0
        while score == 0:
            for i in range(6):
                self.environment['dice'][i] = np.random.randint(1, 6)
            score = farkel_score(*self.environment['dice'])

        return self.get_observable_environment()

    def render(self, mode='human'):
        print(self.observable_environment)

    def close(self):
        self.reset()


def main1():

    env = FarkelEnv()

    print("Learning")

    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    print("Done Learning")

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

def main2():

    env = FarkelEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

if __name__ == "__main__":

    main1()
