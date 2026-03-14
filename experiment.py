import time
import numpy as np
from datetime import datetime

class Experiment:

    def __init__(
        self,
        agent,
        env,
        eval_env,
        eval_episodes,
        total_timesteps,
        eval_interval_timesteps,
        update_after: int = 0,
        max_episodes=-1,
        save_checkpoints: bool = False,
        save_interval_timesteps: int = 1000,
        checkpoint_save_dir: str = "checkpoints",
        env_name: str = None,
        agent_name: str = None,
    ):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.eval_env.mointor = False

        self.timesteps_since_last_eval = 0
        self.timesteps_elapsed = 0

        self.eval_episodes = eval_episodes
        self.total_timesteps = total_timesteps
        self.eval_interval_timesteps = eval_interval_timesteps
        self.update_after = update_after
        self.max_episodes = max_episodes
        self.save_checkpoints = save_checkpoints
        self.save_interval_timesteps = save_interval_timesteps if save_checkpoints else None
        self.checkpoint_save_dir = checkpoint_save_dir
        self.env_name = env_name
        self.agent_name = agent_name

        self.timesteps_since_last_checkpoint = 0
        self.train_episodes = 0
        
        self.train_ep_return = []
        self.train_ep_steps = []
        self.timesteps_at_eval = []
        self.eval_ep_return = []
        self.eval_ep_steps = []

        self.info = {}
        self.train_time = 0.0
        self.eval_time = 0.0
    
    def run(self):
        start_run = time.time()
        print(f"Starting experiment at: {datetime.now()}")

        self.eval_time += self.eval()
        self.timesteps_at_eval.append(self.timesteps_elapsed)


        i = 0
        while self.timesteps_elapsed < self.total_timesteps and \
                (self.train_episodes < self.max_episodes if
                 self.max_episodes > 0 else True):

            # Run the training episode and save the relevant info
            ep_reward, ep_steps, train_time = self.run_episode_train()
            self.train_ep_return.append(ep_reward)
            self.train_ep_steps.append(ep_steps)
            self.train_time += train_time
            print(f"=== Train ep: {i}, r: {ep_reward}, n_steps: {ep_steps}, " +
                  f"elapsed: {train_time}")
            i += 1
        
        self.eval_time += self.eval()
        self.timesteps_at_eval.append(self.timesteps_elapsed)

        end_run = time.time()
        print(f"End run at time {datetime.now()}")
        print(f"Total time taken: {end_run - start_run}")
        print(f"Training time: {self.train_time}")
        print(f"Evaluation time: {self.eval_time}")

        self.info["eval_episode_rewards"] = np.array(self.eval_ep_return)
        self.info["eval_episode_steps"] = np.array(self.eval_ep_steps)
        self.info["timesteps_at_eval"] = np.array(self.timesteps_at_eval)
        self.info["train_episode_steps"] = np.array(self.train_ep_steps)
        self.info["train_episode_rewards"] = np.array(self.train_ep_return)
        self.info["train_time"] = self.train_time
        self.info["eval_time"] = self.eval_time
        self.info["total_train_episodes"] = self.train_episodes

    def run_episode_train(self):
        self.agent.reset()
        self.train_episodes += 1

        episode_rewards = []
        start = time.time()
        episode_return = 0.0
        episode_steps = 0

        state, _ = self.env.reset()
        done = False
        if self.timesteps_elapsed < self.update_after:
            action = self.env.action_space.sample()
        else:
            action = self.agent.sample_action(state)
        
        while not done:
            if self.timesteps_since_last_eval >= self.eval_interval_timesteps:
                self.eval_time += self.eval()
                self.timesteps_at_eval.append(self.timesteps_elapsed)
            
            next_state, reward, done, info = self.env.step(action)
            episode_return += reward
            episode_steps += 1
            episode_rewards.append(reward)
            
            if self.env.steps_per_episode <= 1:
                done_mask = 0
            else:
                if episode_steps <= self.env.steps_per_episode and done and not info.get("steps_exceeded", False):
                    done_mask = 0
                else:
                    done_mask = 1
            
            if self.timesteps_elapsed < self.update_after:
                self.agent.add_to_replay(state, action, reward, next_state, done_mask)
            else:
                self.agent.update(state, action, reward, next_state, done_mask)
            
            if not done:
                if self.timesteps_elapsed < self.update_after:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.sample_action(next_state)
            
            state = next_state
            self.timesteps_elapsed += 1
            self.timesteps_since_last_eval += 1

            self.timesteps_since_last_checkpoint += 1
        
        self.agent.on_episode_end()
        end = time.time()
        return episode_return, episode_steps, (end - start)


    def eval(self):
        self.timesteps_since_last_eval = 0
        self.agent.eval()

        temp_rewards_per_episode = []
        episode_steps = []
        eval_session_time = 0.0

        for i in range(self.eval_episodes):
            eval_start_time = time.time()
            episode_reward, num_steps = self.run_episode_eval()
            eval_end_time = time.time()

            temp_rewards_per_episode.append(episode_reward)
            episode_steps.append(num_steps)

            eval_elapsed_time = eval_end_time - eval_start_time
            eval_session_time += eval_elapsed_time
            # Display the offline episodic return
            print("=== EVAL ep: " + str(i) + ", r: " +
                  str(episode_reward) + ", n_steps: " + str(num_steps) +
                  ", elapsed: " +
                  time.strftime("%H:%M:%S", time.gmtime(eval_elapsed_time)))
        
        self.eval_ep_return.append(temp_rewards_per_episode)
        self.eval_ep_steps.append(episode_steps)
        self.eval_time += eval_session_time
        self.agent.train()

        return eval_session_time

    def run_episode_eval(self):
        state, _ = self.eval_env.reset()
        episode_return = 0.0
        episode_steps = 0
        done = False
        action = self.agent.sample_action(state)
        while not done:
            next_state, reward, done, _ = self.eval_env.step(action)
            episode_return += reward
            episode_steps += 1
            state = next_state
            if not done:
                action = self.agent.sample_action(next_state)
        self.agent.on_episode_end()
        return episode_return, episode_steps