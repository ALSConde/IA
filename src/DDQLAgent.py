import numpy as np
import random
import tensorflow as tf
from collections import deque
from DQLAgent import DQLAgent


class DDQLAgent(DQLAgent):
    def __init__(self, state_size, action_size, game):
        super().__init__(state_size, action_size, game)
        self.target_model = self._build_model()
        self.update_target_model()

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = self.model.predict(state)

            if reward not in [-0.1, -1]:  # end of game or invalid move
                target[0][action] = reward
            else:
                # Double DQN part: use main model to choose an action and target model to generate Q value for that action
                best_action = np.argmax(self.model.predict(next_state)[0])
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[best_action]

            self.model.train_on_batch(state, target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # Atualizar o modelo alvo para ser igual ao modelo
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.game.available_actions())
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, fp):
        self.model.save(fp, overwrite=True, save_format="tf")

    def load(self, fp):
        self.model = tf.keras.models.load_model(fp)
