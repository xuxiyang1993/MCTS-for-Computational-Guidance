
# Copyright Xuxi Yang 2018.
# https://xuxiyang1993.github.io/
# the code to implement Monte Carlo Tree Search algorithm for aircraft Computational Guidance algorithm



import copy
import hashlib
import math
import os
import random
import sys
import itertools
import numpy as np
import pygame
from pygame.locals import *

window_position = (10, 40)  # position of the window
os.environ['SDL_VIDEO_WINDOW_POS'] = str(window_position[0]) + "," + str(window_position[1])
pygame.init()

# white for screen, black, red, green for font
white = (255, 255, 255)
black = (0, 0, 0)
red = (200, 0, 0)
green = (0, 200, 0)

# size of the map
size = width, height = 800, 800
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
gameIcon = pygame.image.load('images/intruder.png')  # icon of the window
pygame.display.set_icon(gameIcon)
pygame.display.set_caption('Monte Carlo Tree Search Agent', 'Spine Runtime')

# exploration-exploitation term, better keep fixed
SCALAR = 1 / math.sqrt(2.0)


intruder_size = 80  # the number of intruder aircraft (red aircraft), better to set it between 10~100
state_size = intruder_size * 4 + 5 + 2  # the dimension of state
action_size = 3
RADIUS = 16
no_frame = 15  # when makes one decision, the simulator will proceed 15 steps
tick = 30
min_dist = 128  # when generating new aircraft, intruder shouldn't be too close to ownship
flexibility_angle = 2  # the ownship can turn 2 degree per time step
np.set_printoptions(precision=2)


class State:
    """
    the class to describe a state
    state is vector includes intruder position, velocity, ownship position, velocity, and goal position
    for example, assume (ix, iy) (ivx, ivy) is intruder pos, vel
    (ox, oy), (ovx, ovy) is ownship pos, vel, and (gx, gy) is goal position
    then state is a vector
    [i1x, i1y, i1vx, i1vy,
     i2x, i2y, i2vx, i2vy,
     ...
     inx, iny, invx, invy,
     ox, oy, ovx, ovy, o_\phi
     gx, gy]
    which is 4*n + 5 + 2 dimensional

    """
    def __init__(self, state=np.zeros(state_size, dtype=float), collide_wall=False, collide_intruder=False,
                 reach_goal=False, prev_action=None, depth=0):
        self.state = state  # the vector contains the position, velocity of all aircraft and position of goal

        # -----------------------------------------
        # the following three binary variables are used to decide if this state is terminal state
        self.collide_wall = collide_wall  # if this state is a collision state
        self.collide_intruder = collide_intruder   # if this state is a collision state
        self.reach_goal = reach_goal  # if this state is a goal state
        # -----------------------------------------

        # the action leads the agent to this state should be stored, so that when a state has max value, we should
        # know which action to take so that we can go to this state.
        self.prev_action = prev_action
        self.depth = depth  # record the depth of this state. Search will stop reaching fixed search depth.

    # given current state 'self' and action 'a', what the next state will be?
    def next_state(self, a):
        return PROCEED(self.state, a, self.depth)

    # decide if this state is a terminal state
    def terminal(self):
        if self.reach_goal or self.collide_intruder or self.collide_wall or self.depth == search_depth:
            return True
        return False

    # decide the reward of this current state
    def reward(self):
        if self.collide_wall:
            r = 0
        elif self.collide_intruder:
            r = 0
        elif self.reach_goal:
            r = +1
        else:
            r = 1 - self.dist_goal() / 1200.0  # according to estimated value function.
        return r

    @property
    def ownx(self):  # return ownship position x
        return self.state[intruder_size * 4 + 0]

    @property
    def owny(self):  # return ownship position y
        return self.state[intruder_size * 4 + 1]

    @property
    def goalx(self):  # return goal position x
        return self.state[intruder_size * 4 + 5]

    @property
    def goaly(self):  # return goal position y
        return self.state[intruder_size * 4 + 6]

    def dist_goal(self):  # return the distance between ownship and goal
        return metric(self.ownx, self.owny, self.goalx, self.goaly)

    # return the distance to the nearest intruder aircraft
    def dist_intruder(self):
        distance = 5000
        for i in range(intruder_size):
            intrux = self.state[4 * i + 0]
            intruy = self.state[4 * i + 1]
            current_dist = metric(intrux, intruy, self.ownx, self.owny)
            if current_dist < distance:
                distance = current_dist

        return distance

    def __hash__(self):
        this_str = str(self.state) + str(self.collide_wall) + str(self.collide_intruder) + str(self.reach_goal)
        return int(hashlib.md5(this_str.encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):  # to decide if two states are identical
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "prev action: %d, own: (%f, %f), goal: (%f, %f), dist goal: %f, dist intruder: %f depth: %d" \
            % (self.prev_action,
               self.ownx,
               self.owny,
               self.goalx,
               self.goaly,
               self.dist_goal(),
               self.dist_intruder(),
               self.depth)
        return s


class Node:
    """
    node class, used to describe the node object in MCTS algorithm
    """
    def __init__(self, state, parent=None):
        self.visits = 0  # how many times this node has been visited
        self.reward = 0.0  # what is the cumulative reward collected through this node

        # self.reward / self.visits is the average value of this node

        self.state = state  # the state vector associated with this node
        self.children = []  # the children this node has (max 3 children)
        self.parent = parent  # parent of this node
        self.untried_action = [0, 1, 2]  # record which action we have tried

    # add a child to this node
    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    # when get new reward from a terminal state, backpropagate the statistics
    def update(self, reward):
        self.reward += reward  # increase cumulative reward
        self.visits += 1   # increase number of visits by 1

    # if this state has 3 children, it is fully expanded (no untried action).
    def fully_expanded(self):
        if len(self.children) == action_size:
            return True
        return False

    def __repr__(self):
        s = "Node: children: %d; visits: %d; reward: %.4f; p_action: %s, state: (%.2f, %.2f); " \
            "goal: (%.2f, %.2f), dist: %.2f, dist2: %.2f" \
            % (len(self.children),
               self.visits,
               self.reward,
               self.state.prev_action,
               self.state.ownx,
               self.state.owny,
               self.state.goalx,
               self.state.goaly,
               self.state.dist_goal(),
               self.state.dist_intruder())
        return s


def UCTSEARCH(budget, root):
    """
    main tree search algorithm
    :param budget: number of simulations, one simulation is one process from root node to terminal node
    :param root: the root node where we will begin our tree search algorithm.
    :return: the best child with largest value
    """
    for _ in range(int(budget)):
        # if iter % 10000 == 9999:
        #     print("simulation: %d" % iter)
        #     print(root)
        front = TREEPOLICY(root)  # follow tree policy to an unseen node
        reward = DEFAULTPOLICY(front.state)  # follow random policy from this unseen node to a terminal node
        BACKUP(front, reward)  # backpropagate this reward info)
    return BESTCHILD(root, 0)


def TREEPOLICY(node):
    """
    tree policy
    :param node: from current code, follow tree policy to an unseen node
    :return: return a node this is unseen before
    """

    while node.state.terminal() == False:
        # a hack to force 'exploitation' in a game where there are many options,
        # and you may never/not want to fully expand first

        # if len(node.children) == 0:
        #     return EXPAND(node)
        # elif random.uniform(0, 1) < .5:
        #     node = BESTCHILD(node, SCALAR)
        # else:
        #     if node.fully_expanded() == False:
        #         return EXPAND(node)
        #     else:
        #         node = BESTCHILD(node, SCALAR)

        # we won't follow tree policy when this node is not fully expanded
        # a hack to force 'exploration'
        if len(node.children) < 3:
            return EXPAND(node)
        # if this node is fully expanded, return its best children (balance exploration & exploitation)
        else:
            node = BESTCHILD(node, SCALAR)
    return node


def EXPAND(node):
    """
    choose a new child from current node
    :param node: the current node we are considering
    :return: the new child
    """
    random_action = random.choice(node.untried_action)  # select an action from the untried action of this node
    node.untried_action.remove(random_action)   # removed the selected action
    new_state = node.state.next_state(random_action)  # given current state, action, decide the next action
    node.add_child(new_state)  # add this new child to the node
    return node.children[-1]


# current this uses the most vanilla MCTS formula it is worth
# experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node, scalar):
    """
    return the best child of current node (balance exploration & exploitation)
    :param node: current considering node
    :param scalar: small number encourage exploitation, large number encourage exploration
    :return: return the best child
    """
    bestscore = -10
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits  # mean reward
        explore = math.sqrt(2 * math.log(node.visits) / float(c.visits))  # formula
        score = exploit + scalar * explore  # UCT value
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:  # this is the best child
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        print("OOPS: no best child found, probably fatal")
    if len(bestchildren) > 1:  # if there are multiple best children
        for node in bestchildren:
            if node.state.prev_action == 1:
                return node
    if scalar == 0:
        # for n in node.children:
        #     if n.state.prev_action == 0:
        #         print('Action <right> has mean value ', round(n.reward/n.visits, 5))
        #     elif n.state.prev_action == 1:
        #         print('Action < str > has mean value ', round(n.reward/n.visits, 5))
        #     elif n.state.prev_action == 2:
        #         print('Action <left > has mean value ', round(n.reward/n.visits, 5))

        r = random.choice(bestchildren)
        # if r.state.prev_action == 0:
        #     print('We are selecting action <right>')
        # elif r.state.prev_action == 1:
        #     print('We are selecting action <straight>')
        # elif r.state.prev_action == 2:
        #     print('We are selecting action <left>')
        # print('--------------------------------------------------------------')
        return r
    return random.choice(bestchildren)


def DEFAULTPOLICY(state):
    """
    random policy
    :param state: from this state, run simulation with random action
    :return: final reward after reaching the final state
    """
    while state.terminal() == False:  # simulate until terminal state
        random_action = random.randint(0, 3)
        state = state.next_state(random_action)
        # print(state)
    return state.reward()


def BACKUP(node, reward):
    """
    backpropagation
    :param node: backpropagate from current node
    :param reward: back up reward
    :return:
    """
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent  # change to its parent and keep backing up
    return


def PROCEED(s, a, d):
    """
    given current state, action, depth, decide the next state
    :param s: current state
    :param a: current action
    :param d: current depth
    :return: next state
    """
    state = copy.deepcopy(s)  # we will change value in state
    # initialize three binary variables
    collide_wall = False
    collide_intruder = False
    reach_goal = False

    delta_direction = (a - 1) * flexibility_angle  # change of heading angle

    for _ in range(no_frame):  # keep updating for no_frame frames
        for i in range(intruder_size):  # updating intruder aircraft position
            # i_x += i_vx
            # i_y += i_vy
            state[4 * i + 0] += state[4 * i + 2]
            state[4 * i + 1] += state[4 * i + 3]

        # update ownship heading angle, velocity, and position
        direction = state[intruder_size * 4 + 4] + delta_direction  # direction in degree
        rad = direction * math.pi / 180
        ownvx = -2 * math.sin(rad)
        ownvy = -2 * math.cos(rad)

        state[intruder_size * 4 + 2] = ownvx
        state[intruder_size * 4 + 3] = ownvy
        state[intruder_size * 4 + 4] = direction

        state[intruder_size * 4 + 0] += ownvx
        state[intruder_size * 4 + 1] += ownvy

        # current ownship position and goal position
        ownx = state[intruder_size * 4 + 0]
        owny = state[intruder_size * 4 + 1]
        goalx = state[intruder_size * 4 + 5]
        goaly = state[intruder_size * 4 + 6]

        # check if the ownship flies out of the map
        if ownx < 8 or ownx > width - 8 or owny < 8 or owny > height - 8:
            # collide with walls
            collide_wall = True
            # next state who cares

        # check if there is any collision with any intruder aircraft
        for i in range(intruder_size):
            x = state[4 * i + 0]
            y = state[4 * i + 1]
            if metric(x, y, ownx, owny) < 32:
                collide_intruder = True
                # who cares next state?

        # check if the aircraft reaches goal state
        if metric(ownx, owny, goalx, goaly) < 32:
            reach_goal = True
            # Hooray!

    # return next state, a new vector, collision information, which action leads to this state
    # and depth increases one
    return State(state, collide_wall, collide_intruder, reach_goal, a, d + 1)


# distance between point (x1,y1) and point (x2,y2)
def metric(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def time_display(count):
    """
    display current time step at top left corner
    :param count: curren time step to display
    :return:
    """
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("time step: " + str(count), True, black)
    screen.blit(text, (5, 0))


def collision_wall(count):
    """
    display how many times the aircraft flies out of map
    :param count:
    :return:
    """
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Collision with wall: " + str(count), True, black)
    screen.blit(text, (5, 30))


def collision_intruder(count):
    """
    display how many collisions with intruder aircraft
    :param count:
    :return:
    """
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Conflict with intruder: " + str(count), True, red)
    screen.blit(text, (5, 60))


def collision_goal(count):
    """
    display number of reached goals
    :param count:
    :return:
    """
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Goal: " + str(count), True, green)
    screen.blit(text, (5, 90))


class DroneSprite(pygame.sprite.Sprite):
    """
    the drone sprite used to describe the ownship object
    """
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load('images/drone.png')
        self.rect = self.src_image.get_rect()
        self.image = self.src_image
        self.position = position  # initial position of the ownship
        self.speed = 2  # speed of ownship: 2 pixel per time step
        self.direction = 0  # initial heading angle
        self.rad = 0
        vx = -self.speed * math.sin(self.rad)
        vy = -self.speed * math.cos(self.rad)
        self.velocity = (vx, vy)  # initiate velocity
        # radius of this object, if distance of two object is larger than the sum of
        # radius of two objects, they are colliding to each other
        self.radius = RADIUS
        self.delta_direction = 0

        # statistics of this ownship: number of collisions, number of reached goals
        self.collision_intruder = 0
        self.collision_wall = 0
        self.collision_goal = 0

    def update(self, deltat):
        self.direction += self.delta_direction  # new heading angle
        self.direction %= 360  # keep it between (0, 360)
        self.rad = self.direction * math.pi / 180
        vx = -self.speed * math.sin(self.rad)
        vy = -self.speed * math.cos(self.rad)
        self.velocity = (vx, vy)  # new velocity

        x = self.position[0] + vx
        y = self.position[1] + vy
        self.position = (x, y)  # new position
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position  # draw


class PadSprite(pygame.sprite.Sprite):
    """
    object used to describe the intruder aircraft
    """
    def __init__(self, position, speed, direction):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load('images/intruder.png')
        self.image = self.src_image
        self.rect = self.src_image.get_rect()
        self.position = position  # initial position
        self.speed = speed  # initial speed
        self.direction = direction  # initial heading angle
        self.rad = self.direction * math.pi / 180
        vx = -self.speed * math.sin(self.rad)
        vy = -self.speed * math.cos(self.rad)
        self.velocity = (vx, vy)  # initial velocity
        self.radius = RADIUS  # radius

    def update(self):
        x, y = self.position
        x += self.velocity[0]
        y += self.velocity[1]
        self.position = (x, y)  # update position
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position  # draw


class GoalSprite(pygame.sprite.Sprite):
    """
    object used to describe the goal
    """
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/goal.png')
        self.rect = self.image.get_rect()
        self.position = position  # initial position
        self.radius = 16
        self.rect = self.image.get_rect()
        self.rect.center = self.position

    def update(self):
        self.rect.center = self.position


def get_state(pads, own, goal):
    """
    get the current state from the simulator, used to make decision
    :param pads: list of intruder aircraft object
    :param own: ownship object
    :param goal: goal object
    :return: a vector
    """
    state_list = []
    for i in range(intruder_size):
        # add (i_x, i_y, i_vx, i_vy) to the state for each intruder
        state_list.append(pads[i].position[0])
        state_list.append(pads[i].position[1])
        state_list.append(pads[i].velocity[0])
        state_list.append(pads[i].velocity[1])
    for i in range(1):
        # add ownship information
        state_list.append(own.position[0])
        state_list.append(own.position[1])
        state_list.append(own.velocity[0])
        state_list.append(own.velocity[1])
        state_list.append(own.direction)
    # add goal information
    state_list.append(goal.position[0])
    state_list.append(goal.position[1])

    return np.array(state_list)


# distance between two sprites
def dist(a, b):
    dx = a.position[0] - b.position[0]
    dy = a.position[1] - b.position[1]
    return math.sqrt(dx * dx + dy * dy)


def reset_intruder(intruder):
    """
    reset a intruder when necessary, and intruder should not be too close to ownship
    :param intruder: intruder aircraft we want to reset its position, when it flies
    out of map, or collide into ownship, or at the beginning of a episode
    :return:
    """
    while dist(ownship, intruder) < min_dist:  # new position should be large than min_dist
        intruder.position = (random.random() * width,
                             random.random() * height)
    # speed is in range (1,2)
    intruder.speed = random.random() + 1
    # random heading angle
    intruder.direction = random.random() * 360
    intruder.rad = intruder.direction * math.pi / 180
    vx = -intruder.speed * math.sin(intruder.rad)
    vy = -intruder.speed * math.cos(intruder.rad)
    intruder.velocity = (vx, vy)


pads = []  # the list of all the intruder aircraft
# generate a goal not too close to the boundary of map
goal = GoalSprite((random.random() * (width - 200) + 100,
                   random.random() * (height - 200) + 100))

# generate many intruder aircraft
for _ in range(intruder_size):
    # generate each intruder with random position, speed from 1 to 2, random heading angle
    pads.append(PadSprite((random.random() * width,
                           random.random() * height),
                          random.random() + 1,
                          random.random() * 360))

rect = screen.get_rect()
ownship = DroneSprite(rect.center)  # generate a ownship in the center
car_group = pygame.sprite.RenderPlain(ownship)  # ownship group
pad_group = pygame.sprite.RenderPlain(*pads)  # intruder group
goal_group = pygame.sprite.RenderPlain(goal)  # goal group

budget = 800  # number of simulations
search_depth = 3  # search depth: can be 2, 3, 4
time_step = 0  # time step shown at top left corner
simulate = True  # simulate = True will makes the simulator keep running

import time
current_time_list = []  # used to track the running time of the algorithm

while simulate:
    time_step += 1

    # get the current state
    current_state = get_state(pads, ownship, goal)
    # make current state a node
    current_node = Node(State(current_state))

    # record the time before the algorithm running
    current_time = int(round(time.time() * 1000))
    # get an action using MCTS
    action = UCTSEARCH(budget, current_node).state.prev_action
    # record the time after the algorithm running
    time_after = int(round(time.time() * 1000))
    # record the running time
    current_time_list.append(time_after - current_time)

    # the change of heading angle, using this the ownship can update pos, vel
    ownship.delta_direction = (action - 1) * flexibility_angle

    # can be used to control ownship using keyboard, currently no use
    for event in pygame.event.get():
        if not hasattr(event, 'key'):
            continue
        down = event.type == KEYDOWN
        if event.key == K_RIGHT:
            ownship.k_right = down * -4
        elif event.key == K_LEFT:
            ownship.k_left = down * 4
        elif event.key == K_ESCAPE:
            sys.exit(0)

    # run simulator 15 time step
    for _ in range(no_frame):
        deltat = clock.tick(tick)
        screen.fill((255, 255, 255))

        # check if the ownship flies out of map
        if ownship.position[0] < 8 or ownship.position[0] > width - 8 \
                or ownship.position[1] < 8 or ownship.position[1] > height - 8:
            # collide with walls
            collide_wall = True
            ownship.collision_wall += 1
            # regenerate a ownship
            ownship.position = (random.random() * 300 + 400,
                                random.random() * 200 + 300)
            ownship.direction = random.random() * 360

        for intruder in pad_group:
            x = intruder.position[0]
            y = intruder.position[1]
            # check if there is collision with each intruder aircraft
            # if pygame.sprite.collide_circle(ownship, intruder):
            if dist(ownship, intruder) < ownship.radius + intruder.radius:
                collide_intruder = True
                ownship.collision_intruder += 1
                reset_intruder(intruder)
            # if the intruder flies out of map, regenerate one
            elif x > width - 4 or x < 4 or y > height - 4 or y < 4:
                intruder.position = ownship.position
                reset_intruder(intruder)

        # check if ownship reaches goal state.
        # if pygame.sprite.collide_circle(ownship, goal):
        if dist(ownship, goal) < ownship.radius + goal.radius:
            # start a new episode
            collide_goal = True
            ownship.collision_goal += 1
            ownship.position = (750, 750)
            ownship.direction = 45
            ownship.rad = ownship.direction * math.pi / 180
            vx = -ownship.speed * math.sin(ownship.rad)
            vy = -ownship.speed * math.cos(ownship.rad)
            ownship.velocity = (vx, vy)
            for intruder in pad_group:  # reset all intruders
                reset_intruder(intruder)
            goal.position = (random.random() * (width - 200) + 100,
                             random.random() * (height - 200) + 100)

        # update and draw all objects
        car_group.update(deltat)
        pad_group.update()
        goal_group.update()

        pad_group.draw(screen)
        car_group.draw(screen)
        goal_group.draw(screen)

        time_display(time_step)
        collision_wall(ownship.collision_wall)
        collision_intruder(ownship.collision_intruder)
        collision_goal(ownship.collision_goal)

        pygame.display.flip()

    if time_step == 10800:
        print('budget: ', budget)
        print('search depth: ', search_depth)
        print('wall: ', ownship.collision_wall)
        print('intruder: ', ownship.collision_intruder)
        print('goal: ', ownship.collision_goal)

        ownship.collision_wall = 0
        ownship.collision_intruder = 0
        ownship.collision_goal = 0

        simulate = False
