import torch
import inference_utils as utils
import learning.eval
import rewards

class GetAction :
    def __init__(self, num_candidates = 1024, horizon = 4) :

        if not isinstance(num_candidates, int) :
            raise TypeError("num candidates must be an integer!")

        if num_candidates <= 0 :
            raise ValueError("num candidates must be greater than 0!")

        if not isinstance(horizon, int) :
            raise TypeError("horizon must be an integer!")

        if horizon <= 0 :
            raise ValueError("horizon must be greater than 0!")

        self.num_candidates = num_candidates
        self.horizon = horizon
        self.dynamics_predictor = learning.eval.DynamicsPredictor()

    def get_action(self, state: torch.Tensor, patient_object) :

        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor!")

        if state.shape != (utils.DIM_STATE,) :
            raise ValueError(f"State must contain exactly {utils.DIM_STATE} features")

        if not torch.isfinite(state).all() :
            raise ValueError("state contains NaN or inf values!")
            
        state = state.float()

        action_sequences = self.generate_actions()
        results = []
        for sequence in action_sequences :
            state_current = state
            reward = 0.0

            for action in sequence :
                next_state = self.predict_next_state(state_current, action)
                
                rew = self.calculate_reward(next_state, action, patient_object)
                reward += rew
                state_current = next_state
                
            results.append((sequence, reward))

        best_sequence = max(results, key=lambda x: x[1])[0]
        best_action = best_sequence[0]
        return self.make_action_valid(best_action)

    def generate_actions(self) :
        if len(utils.ACTION_BOUNDS) != utils.DIM_ACTION:
            raise ValueError("ACTION_BOUNDS size must match DIM_ACTION")

        actions = torch.rand(self.num_candidates, self.horizon, utils.DIM_ACTION)
        low = torch.tensor([x[0] for x in utils.ACTION_BOUNDS])
        high = torch.tensor([x[1] for x in utils.ACTION_BOUNDS])
        actions = low + actions * (high - low)

        for k in range(self.num_candidates) :
            for h in range(self.horizon) :
                actions[k, h] = self.make_action_valid(actions[k, h])

        return actions

    def predict_next_state(self, state, action):
        next_state = self.dynamics_predictor.next_state(state, action)
        return torch.as_tensor(next_state, dtype= torch.float32)

    def calculate_reward(self, state, action, patient_object) :
        state_reward, _ = rewards.compute_state_reward(state, patient_object)
        action_reward = rewards.compute_action_reward(action)
        return state_reward + action_reward

    @staticmethod
    def make_action_valid(action) :
        action = action.clone()
        action[0] = torch.clamp(action[0], 0.0, 1.0)
        action[4] = torch.clamp(action[4], 0.0, 18.0)
        action[4] = torch.minimum(action[4], action[1] * 0.95)

        max_ti = (60.0 / action[3]) * 0.95
        action[2] = torch.minimum(action[2], max_ti)
        action[5] = torch.minimum(action[5], action[2] * 0.95)

        return action

    def __str__(self):
        return f"running is done .***********"