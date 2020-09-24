import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

class GolfCardGameEnv(gym.Env):

    def __init__(self):
        self.state = None
        self.steps_beyond_done = None

        # Setup game specific items
        self.steps_taken = None
        self.number_of_players = 2
        self.discard_pile = None
        self.stage_card = {'p':11}
        self.cards = None
        self.is_inputing = None
        self.players_turn = None
        self.old_players_turn = None
        self.distinct_cards = None
        self.hands = None
        self.is_end_of_game = None
        self.cards, self.distinct_cards = self.get_cards()

        self.outputs: list(str) = [
            'flip 00',
            'flip 01',
            'flip 10',
            'flip 11',
            'flip 20',
            'flip 21',
            'draw discard',
            'draw deck',
            'replace 00',
            'replace 01',
            'replace 10',
            'replace 11',
            'replace 20',
            'replace 21',
            'discard stage'
        ]

        self.inputs = {
            'discard points': {'low':-2,'high':10},
            'stage points': {'low':-2,'high':11}
        }
        for player in range(self.number_of_players):
            for col in range(3):
                for row in range(2):
                    i = str(player)+str(col)+str(row)
                    self.inputs[i+' points'] = {'low':-2,'high':11}
        for d in self.distinct_cards:
            self.inputs[d+' amount in discard'] = {'low':0,'high':4}
        
        self.reset()

        inputs_low: list(float) = []
        for i in self.inputs:
            inputs_low.append(self.inputs[i]['low'])
        inputs_high: list(float) = []
        for i in self.inputs:
            inputs_high.append(self.inputs[i]['high'])
        self.num_obervations: int = len(self.inputs)
        inputs_np_high = np.array(inputs_high, dtype=np.float32)
        inputs_np_low = np.array(inputs_low, dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(inputs_np_low, inputs_np_high, dtype=np.float32)
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.outputs))
        self.seed()

    def deal_hands(self):
        hands = []
        for player in range(self.number_of_players):
            hands.append([])
            for col in range(3):
                hands[player].append([])
                for row in range(2):
                    top_card = self.cards.pop()
                    hands[player][col].append(top_card)
        return hands
        

    def get_cards(self):
        suits = ['S','D','C','H']
        distinct_cards = []
        cards = [] # {'id':'AS','p':1,'vis':False}
        for i in range(9):
            for s in suits:
                cards.append({'id':str(i+2)+s,'p':2+i,'vis':False}) # 2 through 10
            distinct_cards.append(str(i+2))
        for s in suits:
            cards.append({'id':'K'+s,'p':0,'vis':False})
        distinct_cards.append('K')
        for s in suits:
            cards.append({'id':'Q'+s,'p':10,'vis':False})
        distinct_cards.append('Q')
        for s in suits:
            cards.append({'id':'J'+s,'p':10,'vis':False})
        distinct_cards.append('J')
        for s in suits:
            cards.append({'id':'A'+s,'p':1,'vis':False})
        distinct_cards.append('A')
        for i in range(2):
            cards.append({'id':'W'+str(i),'p':-2,'vis':False})
        distinct_cards.append('W')
        self.shuffle_cards(cards)
        return cards, distinct_cards
    
    def shuffle_cards(self,cards):
        random.shuffle(cards)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        if self.players_turn != 0:
            print('ERROR wo38iru')

        # 1) Perform action
        is_invalid = self.take_players_turn(action)

        # 2) Get Done
        done = is_invalid == True or self.is_end_of_game == True or len(self.cards) < 5 # todo change this

        # 3) Get reward
        reward = 0
        if not done:
            reward = 0 # try with 1? or if they get 0 in a column, then give reward
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            if is_invalid == False:
                myscore = self.get_score(0)
                for i in range(self.number_of_players-1):
                    other_score = self.get_score(i+1)
                    if other_score < myscore:
                        reward = 0 # I lost
                    else:
                        reward = (60 - myscore)*10 # TODO do this differently for more players
                self.print_hands()
                print('Game ended successfully',myscore)
            else:
                reward = 0 # don't be invalid
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        # 4) Take other players turns
        while self.players_turn != 0:
            is_invalid = self.take_players_turn()
            if is_invalid:
                print('ERROR 93284235325r823')
                done = True
                reward = 0
                self.players_turn = 0
                self.steps_beyond_done = 0

        # 5) Set state
        self.set_state()

        return np.array(self.state), reward, done, {}
    
    def find_and_swap(self, card_id, destination):
        card_to_swap = None
        found_card = None

        #print('Finding',card_id,'and putting it',destination)

        if destination == 'discard':
            card_to_swap = self.discard_pile.pop()
        elif destination == 'stage':
            card_to_swap = self.stage_card
            self.stage_card = None
        else: #assume this is a location like 120
            for i,player in enumerate(self.hands):
                for j,col in enumerate(player):
                    for k,card in enumerate(col):
                        full_id = str(i)+str(j)+str(k)
                        if full_id == destination:
                            card_to_swap = card

        #print('card_to_swap ',card_to_swap)

        if card_to_swap:
            card_to_swap['vis'] = False
            # hands
            for i,player in enumerate(self.hands):
                for j,col in enumerate(player):
                    for k,card in enumerate(col):
                        if card['vis'] == False and card['id'] == card_id:
                            found_card = card
                            #print('found_card in hands ',found_card)
                            found_card['vis'] = True
                            #print('setting hands card to swap ',self.hands[i][j][k])
                            self.hands[i][j][k] = card_to_swap
                            #print('setting hands card to swap ',self.hands[i][j][k])
            # deck
            for i,card in enumerate(self.cards):
                if card['vis'] == False and card['id'] == card_id:
                    found_card = card
                    #print('found_card in deck ',found_card)
                    found_card['vis'] = True
                    #print('setting deck card to swap ',self.cards[i])
                    self.cards[i] = card_to_swap
                    #print('setting deck card to swap ',self.cards[i])

            if found_card:
                if destination == 'discard':
                    self.discard_pile.append(found_card)
                elif destination == 'stage':
                    self.stage_card = found_card
                else:
                    for i,player in enumerate(self.hands):
                        for j,col in enumerate(player):
                            for k,card in enumerate(col):
                                full_id = str(i)+str(j)+str(k)
                                if full_id == destination:
                                    #print('Putting found card in hand')
                                    #print(self.hands[i][j][k])
                                    self.hands[i][j][k] = found_card
                                    #print(self.hands[i][j][k])
            else:
                print('CARD ID IS INVALID ',card_id)

            
        else:
            print('Error 2w038ruir3089')
        #self.print_hands()

    def reset(self):
        #print('reset')
        self.discard_pile = None
        self.stage_card = None
        self.steps_beyond_done = None
        self.cards, self.distinct_cards = self.get_cards()
        #print('len cards',len(self.cards))
        self.hands = self.deal_hands()
        if self.is_inputing is True:
            self.print_hands()
        self.players_turn = random.randint(0,self.number_of_players-1)
        if self.is_inputing is True:
            answer = input('Who is going first? (0-'+str(self.number_of_players-1)+') ')
            self.players_turn = int(answer)
        #print('self.hands',self.hands)
        self.discard_pile = []
        top_card = self.cards.pop()
        top_card['vis'] = True
        self.discard_pile = [top_card]
        if self.is_inputing is True:
            card_id = input('What is the discard pile card? (ex:'+self.get_random_unknown_card()['id']+') ')
            self.find_and_swap(card_id,'discard')
        #print('self.discard_pile',self.discard_pile)
        self.old_players_turn = None
        self.steps_taken = 0
        self.is_end_of_game = None
        #print('---------Player #',self.players_turn,' is starting')
        while self.players_turn != 0:
            is_invalid = self.take_players_turn()
            if is_invalid:
                print('ERROR 93284r823')
                done = True
                self.players_turn = 0

        self.set_state()
        return np.array(self.state)

    def take_players_turn(self, action=None):
        #print('1 take_players_turn ',self.players_turn,' old = ',self.old_players_turn)
        self.steps_taken += 1
        is_invalid = False
        if self.is_inputing is True:
            self.print_hands()
        if self.is_hand_complete():
            #print('2 is_hand_complete ',True)
            self.end_game()
        if self.is_end_of_game:
            if self.is_inputing is True:
                print('Player '+str(self.players_turn)+' your score was '+str(self.get_score()))
            self.next_turn()
        else:
            if self.is_first_turn():
                #print('2 is_first_turn ',True)
                self.flip_two_random_cards()
                if self.is_inputing is True:
                    self.print_hands()
            if action is None:
                action = self.get_smart_valid_action() # todo: make this not random
                if self.is_inputing is True:
                    print('Available actions:')
                    for i in self.get_valid_actions():
                        print(i,self.outputs[i])
                    answer = input('Player '+str(self.players_turn)+' What action will you take? ')
                    action = int(answer)
            else:
                if self.is_inputing is True:
                    answer = input('Player '+str(self.players_turn)+' I think you should '+self.outputs[action]+'. Do you agree? (y/n) ')
                    if answer is not 'y':
                        print('Available actions:')
                        for i in self.get_valid_actions():
                            print(i,self.outputs[i])
                        answer = input('Player '+str(self.players_turn)+' What action will you take? ')
                        action = int(answer)

            #print('3 action = ',action,' ',self.outputs[action])
            end_of_turn, is_invalid = self.perform_action(action)
            #print('8 end_of_turn = ',end_of_turn)
            #print('8 is_invalid ',is_invalid)
            if end_of_turn: 
                self.next_turn()
            else:
                if self.old_players_turn == self.players_turn:
                    is_invalid = True
                    self.next_turn()
                    print("ERROR 3048ru20983u")
                else:
                    self.old_players_turn = self.players_turn
        if self.steps_taken > 2*100:
            print('**************************************************************************taking a lot of time....')
            is_invalid = True
        return is_invalid #is_invalid
    
    def next_turn(self):
        self.old_players_turn = self.players_turn
        self.players_turn += 1
        if self.players_turn >= self.number_of_players:
            self.players_turn = 0
    
    def is_hand_complete(self):
        for col in self.hands[self.players_turn]:
            for card in col:
                if card['vis'] == False:
                    return False
        return True
    
    def get_random_unknown_card(self):
        return random.choice(self.cards)
    
    def get_valid_actions(self):
        valid_actions = []
        for a,o in enumerate(self.outputs):
            if self.is_valid_action(a,False):
                valid_actions.append(a)
        if len(valid_actions) < 1:
            print('ERROR 2083ry3r028')
        return valid_actions

    def get_random_valid_action(self,debug=True):
        #if debug:
            #for v in valid_actions:
                #print('this is a valid random action ',v,' ',self.outputs[v])
        return random.choice(self.get_valid_actions())
    
    def get_smart_valid_action(self):
        valid_actions = self.get_valid_actions()
        # 1) look for matches
        for c,col in enumerate(self.hands[self.players_turn]):
            if col[0]['vis'] and col[1]['vis'] and col[0]['id'][0] == col[0]['id'][0]: # both known and equal
                pass
            elif col[0]['vis'] and col[1]['vis']: # both known and not equal
                if self.stage_card and self.stage_card['id'][0] == col[0]['id'][0]:
                    return self.outputs.index('replace '+str(c)+str(1))
                elif len(self.discard_pile)>0 and self.discard_pile[-1]['id'][0] == col[0]['id'][0]:
                    return self.outputs.index('draw discard')
                elif self.stage_card and self.stage_card['id'][0] == col[1]['id'][0]:
                    return self.outputs.index('replace '+str(c)+str(0))
                elif len(self.discard_pile)>0 and self.discard_pile[-1]['id'][0] == col[1]['id'][0]:
                    return self.outputs.index('draw discard')
            elif col[0]['vis']: # 1 known
                if self.stage_card and self.stage_card['id'][0] == col[0]['id'][0]:
                    return self.outputs.index('replace '+str(c)+str(1))
                elif len(self.discard_pile)>0 and self.discard_pile[-1]['id'][0] == col[0]['id'][0]:
                    return self.outputs.index('draw discard')
            elif col[1]['vis']: # 1 known
                if self.stage_card and self.stage_card['id'][0] == col[1]['id'][0]:
                    return self.outputs.index('replace '+str(c)+str(0))
                elif len(self.discard_pile)>0 and self.discard_pile[-1]['id'][0] == col[1]['id'][0]:
                    return self.outputs.index('draw discard')
            else: # neither known
                pass

        # 2) take small cards
        if len(self.discard_pile)>0 and self.discard_pile[-1]['p']<5:
            return self.outputs.index('draw discard')

        return self.get_random_valid_action()


    def is_valid_action(self,action,debug=True):
        if self.outputs[action].split()[0] == 'flip':
            is_valid =  self.stage_card is None
            if is_valid == False and debug:
                print('You are attempting to flip after drawing')
            if is_valid == False:
                return is_valid
            place = self.outputs[action].split()[1]
            for c,col in enumerate(self.hands[self.players_turn]):
                for r,card in enumerate(col):
                    i = str(c)+str(r)
                    if place == i:
                        is_valid = card['vis'] == False
                        if is_valid == False and debug:
                            print('#'+str(self.players_turn),'You are attempting to flip a card that is already visible')
                        return is_valid
        if self.outputs[action].split()[0] == 'replace':
            is_valid =  self.stage_card is not None
            if is_valid == False and debug:
                print('You are attempting to replace without drawing')
            return is_valid
        if self.outputs[action].split()[0] == 'draw':
            is_valid =  self.stage_card == None
            if is_valid == False and debug:
                print('You are attempting to draw after drawing')
            return is_valid
        if self.outputs[action].split()[0] == 'discard':
            is_valid =  self.stage_card is not None
            if is_valid == False and debug:
                print('You are attempting to discard without drawing')
            return is_valid
        print('COULD NOT VALIDATE ACTION ',action,' ',self.outputs[action])
        return False


    def perform_action(self,action):
        #print('4 perform_action ',action, ' ',self.outputs[action])
        is_invalid = not self.is_valid_action(action,self.is_inputing)
        if self.is_inputing is True and is_invalid:
            print('WARNING THAT ACTION IS INVALID')
        #print('5 is action invalid ',is_invalid)
        end_of_turn = True
        if not is_invalid:

            card_was_hidden_when_discarded = False

            if self.outputs[action].split()[0] == 'draw':
                pile = place = self.outputs[action].split()[1]
                if pile == 'discard':
                    self.stage_card = self.discard_pile.pop()
                if pile == 'deck':
                    self.stage_card = self.cards.pop()
                    self.stage_card['vis'] = True
                end_of_turn = False
            if self.outputs[action].split()[0] == 'discard':
                self.discard_pile.append(self.stage_card)
                self.stage_card = None
            if self.outputs[action].split()[0] == 'flip':
                place = self.outputs[action].split()[1]
                for c,col in enumerate(self.hands[self.players_turn]):
                    for r,card in enumerate(col):
                        i = str(c)+str(r)
                        if place == i:
                            self.hands[self.players_turn][c][r]['vis'] = True
            if self.outputs[action].split()[0] == 'replace':
                place = self.outputs[action].split()[1]
                for c,col in enumerate(self.hands[self.players_turn]):
                    for r,card in enumerate(col):
                        i = str(c)+str(r)
                        if place == i:
                            temp = self.hands[self.players_turn][c][r]
                            self.hands[self.players_turn][c][r] = self.stage_card
                            self.stage_card = None
                            card_was_hidden_when_discarded = temp['vis'] == False
                            temp['vis'] = True
                            self.discard_pile.append(temp)

            
            if self.is_inputing is True:
                if self.outputs[action].split()[0] == 'draw':
                    pile = place = self.outputs[action].split()[1]
                    if pile == 'deck':
                        card_id = input('Player '+str(self.players_turn)+' What card did you draw? (ex:'+self.get_random_unknown_card()['id']+') ')
                        self.find_and_swap(card_id,'stage')
                if self.outputs[action].split()[0] == 'flip':
                    card_id = input('Player '+str(self.players_turn)+' What was the card you flipped over? (ex:'+self.get_random_unknown_card()['id']+') ')
                    place = self.outputs[action].split()[1]
                    self.find_and_swap(card_id,str(self.players_turn)+place)
                if self.outputs[action].split()[0] == 'replace':
                    if card_was_hidden_when_discarded:
                        card_id = input('Player '+str(self.players_turn)+' What card did you just place in the discard pile? (ex:'+self.get_random_unknown_card()['id']+') ')
                        self.find_and_swap(card_id,'discard')
        #print('5 is end of turn ',end_of_turn)
        return end_of_turn, is_invalid

    def is_first_turn(self):
        if self.stage_card:
            return False
        for col in self.hands[self.players_turn]:
            for card in col:
                if card['vis']:
                    return False
        return True
    
    def get_score(self, player=None):
        if player == None:
            player = self.players_turn
        score = 0
        for col in self.hands[player]:
            p0 = col[0]['p']
            p1 = col[1]['p']
            i0 = col[0]['id'][0]
            i1 = col[1]['id'][0]
            if i0 == 'W' and i1 == 'W':
                score += -4
            elif i0 == i1:
                score += 0
            else:
                score += p0 + p1
        return score
    
    def end_game(self):
        # filp all unknown cards
        if self.is_inputing is True:
            print('GAME OVER')
        for player in self.hands:
            for col in player:
                for card in col:
                    card['vis'] = True
        self.is_end_of_game = True
    
    def flip_two_random_cards(self):
        if self.is_inputing is True:
            print('Player '+str(self.players_turn)+' Please flip two cards over.')
            card_loc_0 = input('Player '+str(self.players_turn)+' What was the first card location? 00,01,10,11,20,21? ')
            card_id_0 = input('Player '+str(self.players_turn)+' What was the first card id? (ex:'+self.get_random_unknown_card()['id']+') ')
            card_loc_1 = input('Player '+str(self.players_turn)+' What was the second card location? 00,01,10,11,20,21? ')
            card_id_1 = input('Player '+str(self.players_turn)+' What was the second card id? (ex:'+self.get_random_unknown_card()['id']+') ')
            for c,col in enumerate(self.hands[self.players_turn]):
                for r,card in enumerate(col):
                    if str(c)+str(r) == card_loc_0:
                        self.flip_card(self.players_turn,c,r)
                        #self.print_hands()
                        self.find_and_swap(card_id_0,str(self.players_turn)+card_loc_0)
                        #self.print_hands()
                    if str(c)+str(r) == card_loc_1:
                        self.flip_card(self.players_turn,c,r)
                        #self.print_hands()
                        self.find_and_swap(card_id_1,str(self.players_turn)+card_loc_1)
                        #self.print_hands()
        else:
            random_indexes = random.sample(range(6), 2)
            for i in random_indexes:
                count = 0
                for c,col in enumerate(self.hands[self.players_turn]):
                    for r,card in enumerate(col):
                        if count == i:
                            self.flip_card(self.players_turn,c,r)
                        count += 1

    def flip_card(self,p,c,r):
        if self.hands[p][c][r]['vis']:
            print("Error 203895234")
        else:
            self.hands[p][c][r]['vis'] = True

    def set_state(self):
        state= []
        if len(self.discard_pile) > 0:
            state.append(self.discard_pile[-1]['p'])
        else:
            state.append(11) # unknown
        if self.stage_card is not None:
            state.append(self.stage_card['p'])
        else:
            state.append(11) # unknown
            
        for player in self.hands:
            for col in player:
                for row in col:
                    if row['vis']:
                        state.append(row['p'])
                    else:
                        state.append(11) # unknown
                        
        for d in self.distinct_cards:
            sum_of_d = 0
            for c in self.discard_pile:
                number_on_card = c['id'][0]
                if number_on_card == d:
                    sum_of_d += 1
            state.append(sum_of_d)

        self.state = tuple(state)

    def render(self, mode='human'):

        if self.is_inputing is None:
            print('Hello')
            answer = input('Do you want to input? (y/n) ')
            if answer == 'y':
                print('yay')
                self.is_inputing = True
                self.reset()
            else:
                self.is_inputing = False

        if self.state is None:
            return None

        return None # this might cause problems

    def close(self):
        if self.is_inputing:
            print('bye')
    
    def print_hands(self):
        for i,player in enumerate(self.hands):
            print('Player',i,'----------------------------')
            first_row = ''
            second_row = ''
            for j,col in enumerate(player):
                for k,card in enumerate(col):
                    if k == 0:
                        if card['vis']:
                            first_row += card['id']+' '
                        else:
                            first_row += '??'+' '
                    else:
                        if card['vis']:
                            second_row += card['id']+' '
                        else:
                            second_row += '??'+' '
            print(first_row)
            print(second_row)
        if self.players_turn is not None and self.stage_card is not None:
            print('Player '+str(self.players_turn)+'\'s stageing card = '+self.stage_card['id'])
        if self.discard_pile is not None and len(self.discard_pile) > 0:
            print('Top discard card = '+self.discard_pile[-1]['id'])
                
