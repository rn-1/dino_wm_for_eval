import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PolicyWrapper:
    def __init__(self, model_name: str = "lerobot/diffusion_pusht", n_action_steps: int = None):
        self.policy = DiffusionPolicy.from_pretrained(model_name, local_files_only=True)
        self.policy.eval()
        if n_action_steps is not None:
            # Override how many actions are consumed per inference call.
            # n_action_steps=1 means fresh diffusion inference at every step.
            # The policy generates a full horizon of actions but only serves
            # n_action_steps from the queue before running inference again.
            self.policy.config.n_action_steps = n_action_steps

    def predict(self, observation: torch.Tensor, image: torch.Tensor = torch.zeros(1, 3, 96, 96)) -> torch.Tensor:
        # Truncate state to the dimension the policy was trained with.
        # The WM dataset proprio may have more dims (e.g. 4) than the policy expects (e.g. 2 for PushT).
        expected_state_dim = self._get_expected_state_dim()
        if expected_state_dim is not None:
            observation = observation[..., :expected_state_dim]
        batch = {"observation.state": observation, "observation.image": image}
        return self.policy.select_action(batch)

    def _get_expected_state_dim(self):
        """Return the state dim the policy expects, trying several LeRobot config attribute paths."""
        cfg = self.policy.config
        # LeRobot >=2.x: input_features dict with PolicyFeature values
        if hasattr(cfg, 'robot_state_feature') and cfg.robot_state_feature is not None:
            return cfg.robot_state_feature.shape[0]
        if hasattr(cfg, 'input_features') and cfg.input_features:
            ft = cfg.input_features.get("observation.state")
            if ft is not None:
                return ft.shape[0]
        # LeRobot 1.x: input_shapes dict with plain lists
        if hasattr(cfg, 'input_shapes') and cfg.input_shapes:
            shape = cfg.input_shapes.get("observation.state")
            if shape is not None:
                return shape[0]
        return None

def main():
    # Load the policy
    policy = PolicyWrapper(model_name="lerobot/diffusion_pusht")

    # observation.state: (batch_size, state_dim)
    # select_action manages the n_obs_steps queue internally
    observation = torch.randn(1, 2)  # (batch=1, state_dim=2) — PushT agent position (x, y)

    # Run inference
    actions = policy.predict(observation)
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Sampled actions: {actions}")

if __name__ == "__main__":
    main()