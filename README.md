# commcraft

Implementation of multi agent environemnt for micro-management in StarCraft I based on [gym-starcraft](https://github.com/alibaba/gym-starcraft)

## Installation

1. Install [OpenAI Gym](https://github.com/openai/gym) and its dependencies.

2. Install [TorchCraft](https://github.com/TorchCraft/TorchCraft) and its dependencies. You can skip the torch client part. 

3. Install [torchcraft-py](https://github.com/deepcraft/torchcraft-py) and its dependencies.

4. Install [gym-starcraft](https://github.com/alibaba/gym-starcraft) and its dependencies.

5. Install the package itself:
    ```
    git clone https://github.com/DebangLi/commcraft.git
    cd commcraft
    pip install -e .
    ```

## Contents

rl_based: baseline of reinforcement learning

rule_based: baselines of rule based methods
## Usage
1. Start StarCraft server with BWAPI by Chaoslauncher.

2. Run examples:

    ```
    cd rule_based
    python weakest_agent.py --ip $server_ip --port $server_port 
    ```
    
    The `$server_ip` and `$server_port` are the ip and port of the server running StarCraft. 